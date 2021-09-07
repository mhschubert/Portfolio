import os
import numpy as np
import pandas as pd
import dill as pickle
import json
import time
import torch
import translator
import transformer
import preprocess
from torchtext.legacy import data
from torchtext.legacy import vocab as torchvocab
from torch.optim import Adam


def train_model(model:translator.Translator, datainfo:dict, path:str, epochs:int, optimizer='adam', leng=None):

    if optimizer == 'adam':
        optimizer = translator.NoamOpt(model.d_model, factor=2, warmup=4000,
                                       optimizer=Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                                      eps=1e-9))
    start = time.time()
    total_loss = 0


    path = os.path.join(path, '../..','models/weights/translator')
    os.makedirs(path, exist_ok=True)
    print('starting to train the model...')
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        torch.save(model.state_dict(), os.path.join(path, 'model_weights'))
        for i, batch in enumerate(datainfo['iterator']):
            inp = batch.src
            outp = batch.trgt
            # remove final token as we never see it (so that input output-shapes match)
            dec_pad_mask = transformer.create_pad_mask(outp[:,:-1], outp[:,:-1], datainfo['pad_out'])
            dec_no_peak = transformer.create_no_peak_mask(outp[:,:-1])
            dec_mask = torch.gt((dec_pad_mask + dec_no_peak), datainfo['pad_out'])
            enc_mask = transformer.create_pad_mask(inp, inp, datainfo['pad_in'])
            pred, enc_attns, dec_attns, dec_enc_attns = model(inp, outp[:,:-1], enc_mask, dec_mask)
            optimizer.optimizer.zero_grad()
            #reshape preds so that we have batch first, classes second and word-instances (sequence) last
            #remove Start-Token from target so that shape matches (and we actually only measure loss on true target
            loss = torch.nn.CrossEntropyLoss(pred.contiguous().view(-1,pred.size(-1)), outp[:,1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.float()


            if epoch % 10 == 0:
                print('Current total training time is {} minutes'.format((time.time()-start)//60))
                print('Current loss up to batch {}/{}: {}'.format(i, leng, epoch_loss))
                print('Current total loss in epoch {}/{}: {}'.format(epoch, epochs, epoch_loss if total_loss == 0 else total_loss))

        total_loss += epoch_loss

    print('Total training loss is: {}'.format(total_loss))
    print('Total training time is: {} min'.format((time.time()-start)//60))
    torch.save(model.state_dict(), os.path.join(path, 'model_weights'))

    return model

def predict(model:translator.Translator, datainfo:dict):

    #we have only one batch so meh
    res = None
    tot_loss = 0
    model.eval()
    for i, batch in enumerate(datainfo['iterator']):
        inp = batch.src
        outp = batch.trgt
        # remove final token as we never see it (so that input output-shapes match)
        dec_pad_mask = transformer.create_pad_mask(outp[:, :-1], outp[:, :-1], datainfo['pad_out'])
        dec_no_peak = transformer.create_no_peak_mask(outp[:, :-1])
        dec_mask = torch.gt((dec_pad_mask + dec_no_peak), datainfo['pad_out'])
        enc_mask = transformer.create_pad_mask(inp, inp, datainfo['pad_in'])

        preds, enc_attns, dec_attns, dec_enc_attns = model(inp, outp[:,:-1], enc_mask, dec_mask)
        preds = preds.detach()

        #transform raw to softmax
        loss = torch.nn.CrossEntropyLoss(preds.contiguous().view(-1, preds.size(-1)), outp[:, 1:].contiguous().view(-1))
        tot_loss += loss.float()
        preds = torch.nn.LogSoftmax(preds, dim=preds.size()[-1])
        if res is None:
            res = [preds]
            fin_enc_attns = [enc_attns]
            fin_dec_attns = [dec_attns]
            fin_dec_anc_attns = [dec_enc_attns]
        else:
            res.append(preds)
            fin_enc_attns.append(enc_attns)
            fin_dec_attns.append(dec_attns)
            fin_dec_anc_attns.append(dec_enc_attns)

    print('Total testset loss is {}'.format(round(tot_loss, 5)))

    return res, fin_enc_attns, fin_dec_attns, fin_dec_anc_attns

def get_tokens(preds:list, vocab:torchvocab.Vocab):
    eos = int(vocab.lookup_indices('<EoS>'))
    sos = int(vocab.lookup_indices('<SoS>'))
    pad = int(vocab.lookup_indices('<PAD>'))

    text = []
    for batch in preds:
        tokinds = batch.argmax(dim=-1)
        #last dimension is reduced now -> shape(batchsize, seqlen)
        nz = (tokinds == eos).nonzero()[:,-1]
        nz = nz.unsqueeze(0).transpose(0,1).expand(nz.shape[0], tokinds.shape[1]) #make right shape
        #make indicetensor
        inds = np.arange(0, batch.shape[-2]).unsqueeze(0)
        inds = inds.expand(batch.shape[0], inds.shape[1]) #make shape batchsize, seqlen

        # get all positions with inds smaller than eos and not pad or sos token
        mask = (inds < nz) & (tokinds != sos) & (tokinds != pad)
        for sentence in tokinds.shape[0]:
            tokens = tokinds[sentence, :][mask[sentence, :]]
            tokens = vocab.lookup_tokens(list(tokens))
            text.append(tokens)

    return text

def make_bleu_scores(path, predictions, from_to:str):
    splts = from_to.split(' ')
    src = splts[0]
    trgt = splts[1]
    tokenizer = preprocess.Tokenizer(lang=trgt)
    test = pd.read_csv(os.path.join(path, 'tmp',
                                    'tmp_train_{}_{}_{}.csv'.format(False, src, trgt)),
                       index_col=False, header=None)
    test_trgt = test[trgt].values.tolist()
    test_trgt = [tokenizer.tokenize(sentence) for sentence in test_trgt]
    tester = {'predictions': predictions, 'testdata': test_trgt}
    bleu_score = data.metrics.bleu_score(predictions, test_trgt)
    print('The BLEU score on the test set is: {}'.format(round(bleu_score, 5)))

    return bleu_score, tester

def run(path,from_to, vocabs, epochs = 500, device=None, maxlen=None, retrain=True):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if maxlen is None:
        #set the maxlen to the vocab len for positional encoding
        maxlen = max(vocabs)

    dat, leng, pidx_in, pidx_out, = preprocess.create_dataset(path=path, lang_from_to=from_to, batchsize=500, device=device,
                                                              max_vocab_in=vocabs[0], max_vocab_out=vocabs[1], maxlen=maxlen,
                                                              train=True)
    datainfo = {'iterator': dat,
                'pad_in': pidx_in,
                'pad_out': pidx_out}
    model = translator.setup_model(max_seq_len=maxlen, pidx_in=pidx_in, pidx_out=pidx_out, vocab_size_in=vocabs[0],
                                   vocab_size_out=vocabs[1],d_model=512, n_heads=8, N=6, d_ff=2048,
                                   d_k=None, d_v=None, dropout=0.1, device=device)

    if retrain or os.path.exists(os.path.join(path, '../..','models/weights/translator/model_weights')):
        model = train_model(model=model, datainfo=datainfo, path=path, epochs=epochs, leng=leng)
    else:
        path2 = os.path.join(path, '../..','models/weights/translator')
        model.load_state_dict(torch.load(os.path.join(path2, 'model_weights')))


    testdat, testleng, t_pidx_in, \
    t_pidx_out, src_vocab, trgt_vocab = preprocess.create_dataset(path=path, lang_from_to=from_to, batchsize=None,
                                                                  device=device,max_vocab_in=vocabs[0],
                                                                  max_vocab_out=vocabs[1], maxlen=maxlen,
                                                                  train=False)
    testinfo = {'iterator': testdat,
                'pad_in': t_pidx_in,
                'pad_out': t_pidx_out}
    testres, test_enc_attns, test_dec_attns, test_dec_enc_attns = predict(model, testinfo)
    text = get_tokens(testres, trgt_vocab)
    bleu_score, testdic = make_bleu_scores(path, text, from_to=from_to)

    #save test results
    os.makedirs(os.path.join(path, '../..', 'testresult'), exist_ok=True)
    with open(os.path.join(path, '../..', 'testresult', 'test_pred.json'), 'w') as w:
        json.dump(testdic, w)

    with open(os.path.join(path, '../..', 'testresult', 'enc_attns.p'), 'wb') as w:
        pickle.dump(test_enc_attns, w)

    with open(os.path.join(path, '../..', 'testresult', 'dec_attns.p'), 'wb') as w:
        pickle.dump(test_dec_attns, w)

    with open(os.path.join(path, '../..', 'testresult', 'dec_enc_attns.p'), 'wb') as w:
        pickle.dump(test_dec_enc_attns, w)

    print('done')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '../data/tatoeba/deu_eng'
    from_to = 'deu eng'
    vocabs = [10000,10000]
    epochs = 500
    maxlen = 10000

    run(path, from_to,vocabs,epochs,device,maxlen,retrain=True)