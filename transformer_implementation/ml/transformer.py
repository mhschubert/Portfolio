from __future__ import unicode_literals, print_function, division
import copy
import numpy as np
import math

#ml modules
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as fun
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_pad_mask(query_sequence:torch.Tensor, key_sequence:torch.Tensor, pad_idx:int):

    batchsize, len_query = query_sequence.size()
    batchsize, len_key = key_sequence.size()
    pad_mask = torch.eq(key_sequence, pad_idx).unsqueeze(1)

    pad_mask = torch.as_tensor(pad_mask, dtype=torch.int)

    return pad_mask.expand(batchsize, len_query, len_key)


def create_no_peak_mask(sequence:torch.Tensor):
    batch_size, seq_len = sequence.size()
    no_peak_mask = np.triu(np.ones(shape=(batch_size, seq_len, seq_len)), k=1) #triu because we use .mask_fill
    return no_peak_mask

def clone_layer(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for i in range(N)])

class GaussianRectifier(nn.Module):
    #taken from BERT
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        #positional encodings can be precomputed so we do so here
        enc = torch.zeros(max_seq_len, d_model)
        i = torch.arange(0., max_seq_len).unsqueeze(1)

        divisor = torch.exp(-1*(torch.arange(0, d_model, 2) * math.log(10000) / d_model))
        enc[:,0::2] = torch.sin(i * divisor)
        enc[:,1::2] = torch.cos(i*divisor)
        enc = enc.unsqueeze(0)
        self.register_buffer('penc', enc)

    def forward(self, x):
        x = x + Variable(self.penc[:,:x.size(1)],requires_grad = False)

        if self.dropout is not None:
            return self.dropout(x)
        else:
            return x

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = d_k
        self.device = device
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
        weights = torch.div(torch.matmul(Q,
                                         torch.transpose(K, -1 ,-2)
                                         ),
                            np.sqrt(self.dk))
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool)
            mask = mask.to(self.device)
            weights = weights.masked_fill(mask, -1e9)

        if self.dropout is not None:
            weights = self.dropout(weights)
        attention = self.softmax(weights)
        context = torch.matmul(attention, V)

        return context, attention

class MultiHead(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, device, dropout=0.1):
        super(MultiHead, self).__init__()
        self.dm =d_model
        self.dk = d_k
        self.dv = d_v
        self.heads = n_heads
        #not relevant for testing
        #assert d_model%n_heads == 0 and n_heads*d_v == d_model
        self.device = device

        self.WQ = nn.Linear(d_model, d_k*n_heads)
        self.WK = nn.Linear(d_model, d_k*n_heads)
        self.WV = nn.Linear(d_model, d_v*n_heads)
        self.attention = ScaledDotProductAttention(d_k=self.dk, device=self.device, dropout=dropout)
        self.linear = nn.Linear(d_v*n_heads, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor):
        batch_size = Q.size()[0]
        #dims are batch-size, squence-len, num heads, key-dimensionality

        q = self.WQ(Q).view(batch_size, -1, self.heads, self.dk)
        k = self.WK(K).view(batch_size, -1, self.heads, self.dk)
        v = self.WV(V).view(batch_size, -1, self.heads, self.dv)

        #transpose to get batch-size*num_heads*sequence-len*key-dim
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        #add dimension for n_heads and repeat accordingly in that dimension
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1,1)
        context, attention = self.attention(Q=q, K=k, V=v, mask=mask)
        #make the concat of the context in dim heads and dv via a view
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.heads * self.dv)
        #linear
        return self.linear(context), attention

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = GaussianRectifier()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def forward(self, x:torch.Tensor):
        out = self.l1(x)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.l2(out)
        return out

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.dm = d_model
        self.heads = n_heads
        if dropout is not None:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.l_att = MultiHead(d_model=d_model,d_k=d_k, d_v=d_v, n_heads=n_heads,
                             dropout=dropout, device=device)

        self.l_ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, inputs_enc, mask_enc):

        encoded, attn = self.l_att(Q=inputs_enc, K=inputs_enc, V=inputs_enc,
                                   mask=mask_enc)

        if self.dropout1 is not None:
            encoded = self.dropout1(encoded)
        else:
            encoded = encoded
        #add & norm
        encoded = self.norm1(inputs_enc + encoded)

        #1x1 convolution
        out_enc = self.l_ff(encoded)
        if self.dropout2 is not None:
            out_enc = self.dropout2(out_enc)
        #add & norm
        out_enc = self.norm2(encoded + out_enc)

        return out_enc, attn

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dm = d_model
        self.heads = n_heads
        if dropout is not None:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None
            self.dropout3 = None

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn_dec = MultiHead(d_model=d_model,d_k=d_k, d_v=d_v, n_heads=n_heads,
                             dropout=dropout, device=device)

        self.attn_enc_dec = MultiHead(d_model=d_model,d_k=d_k, d_v=d_v, n_heads=n_heads,
                             dropout=dropout, device=device)

        self.l_ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, inputs_dec, in_mask_dec, outputs_enc, out_mask_enc):
        #first attention head
        out_dec, attn_dec = self.attn_dec(Q=inputs_dec, K=inputs_dec,
                                          V=inputs_dec, mask=in_mask_dec)

        #dropout after first head
        if self.dropout1 is not None:
            out_dec = self.dropout1(out_dec)
        else:
            out_dec = out_dec
        #add & norm
        out_dec = self.norm1(out_dec + inputs_dec)
        #second attention head
        out_dec_enc, attn_dec_enc = self.attn_enc_dec(Q=out_dec, K=outputs_enc, V=outputs_enc, mask=out_mask_enc)
        #dropout
        if self.dropout2 is not None:
            out_dec_enc = self.dropout2(out_dec_enc)

        #add & norm
        out_dec = self.norm2(out_dec_enc + out_dec)
        #feed-forward and add norm
        out_ff = self.l_ff(out_dec)
        if self.dropout3 is not None:
            out_ff = self.dropout2(out_ff)

        out_dec = self.norm3(out_ff + out_dec_enc)

        return out_dec, attn_dec, attn_dec_enc
    
class Encoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, N, device,
                 pad_idx, dropout=0.1, max_seq_len=5000):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.init_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = PositionalEncoder(d_model=d_model, dropout=dropout, max_seq_len=max_seq_len)

        self.layers = clone_layer(EncoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k,
                                               d_v=d_v, n_heads=n_heads, device=device, dropout=dropout), N)

    def forward(self, x, mask_enc):
        out = self.init_embedding(x)
        out = self.pos_embeddings(out)

        #mask_enc = create_pad_mask(x,x, self.pad_idx)

        enc_attns = []
        for layer in self.layers:
            out, attn = layer(out, mask_enc)
            enc_attns.append(attn)

        enc_attns = torch.stack(enc_attns)
        enc_attns = enc_attns.permute([1, 0, 2, 3, 4])
        return out, enc_attns

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, N, device,
                 pad_idx, dropout=0.1, max_seq_len=5000):
        super(Decoder, self).__init__()
        self.pidx = pad_idx
        self.device =device
        self.init_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoder(d_model=d_model, dropout=dropout, max_seq_len=max_seq_len)

        self.layers = clone_layer(DecoderLayer(d_model=d_model, d_ff=d_ff, d_k=d_k,d_v=d_v,
                                               n_heads=n_heads, device=device, dropout=dropout),N)

    def forward(self, dec_in, enc_in, enc_out, dec_mask):

        dec_out = self.init_embed(dec_in)
        dec_out = self.pos_embed(dec_out)

        #dec_pad_mask = create_pad_mask(dec_in, dec_in, self.pidx)
        #dec_no_peak = create_no_peak_mask(dec_in)
        #dec_mask = torch.gt((dec_pad_mask + dec_no_peak), 0)
        enc_mask = create_pad_mask(dec_in, enc_in, self.pidx)

        dec_attns = []
        enc_attns = []

        for i, layer in enumerate(self.layers):
            dec_out, dec_attn, enc_attn = layer(dec_out, dec_mask, enc_out, enc_mask)

            dec_attns.append(dec_attn)
            enc_attns.append(enc_attn)

        dec_attns = torch.stack(dec_attns)
        enc_attns = torch.stack(enc_attns)

        dec_attns = dec_attns.permute([1, 0, 2, 3, 4])
        enc_attns = enc_attns.permute([1, 0, 2, 3, 4])

        return dec_out, dec_attns, enc_attns

class SequenceToSequence(nn.Module):
    #basically like translator predicting sequence b from sequence a
    def __init__(self, vocab_a_size, vocab_b_size, d_model,
                 d_ff, n_heads, N, p_idx_a, p_idx_b, d_k=None, d_v=None, dropout=0.1, max_seq_len=5000):
        super(SequenceToSequence, self).__init__()
        #assert that we follow dimension specifications from attention is all you need
        if d_k is None or d_k is None:
            assert d_model % n_heads == 0
            d_k = d_model // n_heads
            d_v = d_k
        else:
            assert n_heads * d_k == d_model
            assert n_heads * d_v == d_model
            pass
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(vocab_size=vocab_a_size,
                               d_model=d_model, d_ff=d_ff, d_k=d_k,
                               d_v=d_v, n_heads=n_heads, N=N, pad_idx=p_idx_a, device=self.device,
                               dropout=dropout, max_seq_len=max_seq_len)

        self.decoder = Decoder(vocab_size=vocab_b_size,
                               d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v,
                               n_heads=n_heads, N=N, pad_idx=p_idx_b, dropout=dropout,
                               device=self.device, max_seq_len=max_seq_len)

        self.linear_layer = nn.Linear(d_model, vocab_b_size,bias=False)



    def forward(self, inputs_a, inputs_b, enc_mask, dec_mask):
        out_enc, enc_attns = self.encoder(inputs_a, enc_mask)

        out_dec, dec_attns, dec_enc_attns = self.decoder(inputs_b, inputs_a, out_enc, dec_mask)
        out = self.linear_layer(out_dec)
        out = nn.functional.softmax(out, dim=-1)
        return out, enc_attns, dec_attns, dec_enc_attns


if __name__ == '__main__':
    input_a = [[2, 4, 6, 1, 3],
               [2, 4, 6, 1, 3],
               [2, 4, 6, 1, 3],
               [2, 4, 6, 1, 3],
               #[2, 4, 6, 1, 3]
               ]
    #technically it should not matter whether we have these zeros here or not (no-peak mask is used)
    input_b = [[1, 0, 0, 0, 0],
               [1, 3, 0, 0, 0],
               [1, 3, 5, 0, 0],
               [1, 3, 5, 2, 0],
               #[1, 3, 5, 2, 4]
               ]

    input_a = torch.as_tensor(input_a, dtype=torch.int).to(device)
    input_b = torch.as_tensor(input_b, dtype=torch.int).to(device)

    #precalculate mask because it stays fixed
    dec_pad_mask = create_pad_mask(input_b, input_b, 0)
    dec_no_peak = create_no_peak_mask(input_b)
    dec_mask = torch.gt((dec_pad_mask + dec_no_peak), 0)
    enc_mask = create_pad_mask(input_a, input_a, 0)
    print(enc_mask.shape)
    model = SequenceToSequence(vocab_a_size=7, vocab_b_size=8, d_model=256,
                               d_ff=512, d_k=64, d_v=64, n_heads=4, N=4, p_idx_a=0, p_idx_b=0, dropout=None)

    preds, enc_attns, dec_attns, dec_enc_attns = model(input_a, input_b, enc_mask, dec_mask)
    print(preds)
    print(preds.argmax(-1))
    print(preds.shape)





