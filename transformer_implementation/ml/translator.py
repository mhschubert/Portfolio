#implements a tanslator
from __future__ import unicode_literals, print_function, division
#ml modules
import torch
import torch.nn as nn
from transformer import Encoder, Decoder



class NoamOpt:
    "Optim wrapper that implements rate by nlp.seas.harvard.edu/; adapted for torch 1.1.0"

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 1
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p['lr'] = self._rate
    def step(self):
        "Update parameters and rate"
        #torch 1.1.0
        self.optimizer.step()
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        #old way optimizer step after scheduler step
        #self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Translator(nn.Module):

    def __init__(self, vocab_a_size, vocab_b_size, d_model,
                 d_ff, n_heads, N, p_idx_a, p_idx_b, device, d_k=None, d_v=None, dropout=0.1, max_seq_len=5000):
        super(Translator, self).__init__()
        #assert that we follow dimension specifications from attention is all you need
        if d_k is None or d_k is None:
            assert d_model % n_heads == 0
            d_k = d_model // n_heads
            d_v = d_k
        else:
            assert n_heads * d_k == d_model
            assert n_heads * d_v == d_model
            pass
        self.device = device
        self.d_model = d_model
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
        print('translator mask shape')
        print(out_enc.shape)
        print(enc_mask.shape)
        print(inputs_b.shape)
        print(dec_mask.shape)
        out_dec, dec_attns, dec_enc_attns = self.decoder(inputs_b, inputs_a, out_enc, dec_mask)
        out = self.linear_layer(out_dec)
        return out, enc_attns, dec_attns, dec_enc_attns

def setup_model(max_seq_len, pidx_in, pidx_out, vocab_size_in, vocab_size_out,
                d_model=512, n_heads=8, N=6, d_ff=2048, d_k=None, d_v=None, dropout=0.1, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    translator = Translator(vocab_a_size=vocab_size_in, vocab_b_size=vocab_size_out,
                            d_model=d_model, n_heads=n_heads, N=N, d_ff=d_ff, d_k=d_k, d_v=d_v,
                            max_seq_len=max_seq_len, p_idx_a=pidx_in, p_idx_b=pidx_out, dropout=dropout,
                            device=device)

    for p in translator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if device == 'cuda':
        translator.to(torch.device(device))
    return translator