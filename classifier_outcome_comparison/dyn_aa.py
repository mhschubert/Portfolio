#Author: Marcel H. Schubert
'''Script is for large-scale comparison of performance on different featuretypes and their combinations.
It implements two models, a single appraoch using a SVM and the stacked approach suggested in Custodio et al. (2021):
https://www.sciencedirect.com/science/article/abs/pii/S0957417421003079
The target is either gender or age of authors and thus we are in the field of atuhroship analysis/profiling.

Caution: When I say large-scale I mean it: This script trains 4800 models and comapres them in accuracy, f1-score and
on internal similarity when the featuretype stays the same but the target-set changes slightly in the number of authors.'''


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import demoji
import pkg_resources
pkg_resources.require("Scipy==1.7.0")
import scipy
import numpy as np
import pickle
import statistics
import json
import ndjson
import re
import os
import gc
import copy
import multiprocessing as mp
import tensorflow as tf
import sys
import ast
import random
import jsonlines
from utils import ml_utils, generator, evaluations

def compose_model(lower_weights:list , upper_weights:np.ndarray):
    # helper function to reconstruct a trained model from a weight matrix
    # build the model with the trained weights so that we can do relevance-propagation
    inputshape = 0
    conc_out = 0
    models = []
    inputs = []
    for weightm in lower_weights:
        curri =  weightm[0].shape[0]
        curro = weightm[0].shape[-1]
        inputshape+=curri
        conc_out += curro
        tmp = make_logistic(output_shape=curro, functional=True, input_shape=curri, compile=False)
        inputs.append(tmp.get_layer('input'))
        tmp.set_weights(weightm)
        models.append(tmp)

    assert conc_out == upper_weights[0].shape[0]

    out_shape = upper_weights[0].shape[1]

    if upper_weights[0].shape[1] == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    merge_layer = tf.keras.layers.concatenate(models, name='concat')
    out = tf.keras.layers.Dense(out_shape, activation=activation, name='out')(merge_layer)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.get_layer('out').set_weights(upper_weights)



    return model

def make_logistic(output_shape, functional=False, input_shape=None, compile=True):
    # function to create a logistic classifier using tensorflow
    if output_shape==1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    opt = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.5)

    regularizer = tf.keras.regularizers.l2(l2 = 0.01)

    if not functional:
        model = tf.keras.models.Sequential()
        #use delayed building - no input shape specified till we actually build the model (i.e. when we have the data)

        model.add(tf.keras.layers.Dense(output_shape, kernel_regularizer=regularizer,
                                        activation=activation))

    else:
        input = tf.keras.Input(shape=(input_shape,), name='input')
        dense_out = tf.keras.layers.Dense(output_shape, activation=activation, name='output',
                                          kernel_regularizer=regularizer)(input)
        model = tf.keras.Model(inputs=input, outputs=dense_out)

    if compile:
        model.compile(optimizer=opt, loss=loss, metrics='accuracy')




    return model

def make_emoji_vector(featuretyp: str, minTw_gram_auth: list, path: str, id_type: str, subset='train', id_subset=[],
                      vectorizer=None, load_sparse=True):
    # make token feature-matrix for emojis and emoticons

    # get dic for emoji names
    set_ids = {}
    set_ids[featuretyp + ''] = []
    path_parts = ml_utils.split_path_unix_win(path)

    id0 = id_type.split('_')[0]
    model_dir = ml_utils.make_save_dirs(path)
    os.makedirs(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0), exist_ok=True)
    subset_path = os.path.join(*path_parts, featuretyp, str(minTw_gram_auth[0]), id0, subset)
    os.makedirs(os.path.join(subset_path, 'vectorized'), exist_ok=True)
    sparse_path = os.path.join(subset_path, 'vectorized', "vectorized_{}_{}_{}_{}_authors".format(subset, featuretyp,
                                                                                                  minTw_gram_auth[0],
                                                                                                  minTw_gram_auth[2]))
    scaled_path = os.path.join(subset_path, 'vectorized', "vec_scaled_{}_{}_{}_{}_authors".format(subset, featuretyp,
                                                                                                  minTw_gram_auth[0],
                                                                                                  minTw_gram_auth[2]))
    if subset == 'crossval':
        subset = 'train'
        true_train = False
    elif subset == 'train':
        true_train = True
    else:
        true_train = False


    f = open(os.path.join(*path_parts, featuretyp, str(minTw_gram_auth[0]), id0, subset,
                          "{}_{}_{}_{}_authors.ndjson".format(subset,
                                                              featuretyp,
                                                              minTw_gram_auth[0],
                                                              minTw_gram_auth[2])),
             'r', encoding='utf-8')

    # correct emoji count errors
    with open(os.path.join(*path_parts, 'num', str(minTw_gram_auth[0]), id0, subset,
                           "{}_{}_{}_{}_authors.ndjson".format(subset,
                                                               'num',
                                                               minTw_gram_auth[0],
                                                               minTw_gram_auth[2])),
              'r', encoding='utf-8') as n:
        num = ndjson.load(n)
    num = pd.DataFrame.from_dict(num, orient='columns')
    num = num.drop_duplicates(subset=['uID'], keep='first').reset_index(drop=True)
    num.emojis_num = 0
    num.emotic.num = 0
    exist_file = os.path.exists(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])))



    if true_train:
        demojifile = os.path.join(demoji.DIRECTORY, 'codes.json')
        with open(demojifile, 'r') as d:
            demojis = json.load(d)
        # make demojivocab
        vocab = {}
        i = 0
        for emoji in demojis['codes'].values():
            if emoji not in vocab.keys():
                vocab[emoji] = i
                i += 1

        ##add emoticons to vocab
        for emot in [r':\\', r':-/', r':-\\', r':)', r';)', r':-)', r';-)', r':(', r':-(', r':-o', r':o', r'<3']:
            vocab[emot] = i
            i += 1

        vectorizer = CountVectorizer(analyzer=ml_utils.identity_analyzer, vocabulary=vocab)
        scaler = StandardScaler()
        with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                  'vectorizer_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])),
                  'wb') as p:
            pickle.dump(vectorizer, p)


    else:
        if not vectorizer:
            with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'vectorizer_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])),
                      'rb') as p:
                vectorizer = pickle.load(p)

            with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])),
                      'rb') as p:
                scaler = pickle.load(p)
    if not load_sparse:
        exists = False
        id_tester = set()
        while True:
            lines = ml_utils.batch_read(fileobj=f, batchsize=10000, to_json=True)
            if not lines:
                break
            lines = pd.DataFrame.from_dict(lines, orient='columns')
            lines, id_tester = ml_utils.duplicate_uid_tester(lines, id_tester)
            num = ml_utils.reset_num(num, lines)
            if lines.shape[0] == 0:
                # no ids in current lines
                continue
            if id_subset:
                lines = ml_utils.select_uids(lines, id_subset)
                if lines.shape[0] == 0:
                    # no ids in current lines
                    continue
            # save the ids in order
            set_ids[featuretyp + ''].extend(lines['uID'].to_list())
            emoti_sparse = vectorizer.transform(lines['emoticon'].to_numpy())
            emoji_sparse = vectorizer.transform((lines['emoji'].to_numpy()))
            if exists:
                csr_out = scipy.sparse.vstack((csr_out, emoji_sparse + emoti_sparse))
            else:
                csr_out = emoji_sparse + emoti_sparse
                exists = True
        if true_train:
            scaled = scaler.fit_transform(csr_out.toarray())
            with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])),
                      'wb') as p:
                pickle.dump(scaler,p)
        else:
            scaled = scaler.transform(csr_out.toarray())
        np.save(file=scaled_path, arr=scaled)
        #save sparse
        ml_utils.save_sparse(csr_out, sparse_path)
        with open(os.path.join(subset_path, "vectorized","ids_{}_{}_{}_{}_authors".format(subset, featuretyp, minTw_gram_auth[0], minTw_gram_auth[2])),'wb') as w:
            pickle.dump(set_ids, w)
    else:
        csr_out = ml_utils.load_sparse(sparse_path)
        scaled = np.load(scaled_path)
        with open(os.path.join(subset_path, "vectorized","ids_{}_{}_{}_{}_authors".format(subset, featuretyp, minTw_gram_auth[0], minTw_gram_auth[2])),'rb') as w:
            set_ids = pickle.load(w)
    # save fixed num
    with open(os.path.join(*path_parts, 'num', str(minTw_gram_auth[0]), id0, subset,
                           "{}_{}_{}_{}_authors.ndjson".format(subset,
                                                               'num',
                                                               minTw_gram_auth[0],
                                                               minTw_gram_auth[2])),
              'w', encoding='utf-8') as n:
        ndjson.dump(num.to_dict(orient='records'), n)

    processors = [vectorizer, scaler]
    processors = [0, 0]
    return processors, scaled, set_ids

def make_tfidf(featuretyp:str, minTw_gram_auth:list, path:str, id_type:str, subset ='train', hash =True, id_subset=[],
               vectorizers = None, trans_in=None,
               load_sparse=True, base=False):
    # make tfidif-feature matrix for everything (e.g., char-based ngrams, word-based ngrams, emojicons) except numerical features
    minTW = minTw_gram_auth[0]
    numAuthors = minTw_gram_auth[2]
    gram_files = {}
    id0 = id_type.split('_')[0]
    set_ids = {}
    model_dir = ml_utils.make_save_dirs(path)
    vectorizer_dic = {}

    path_parts = ml_utils.split_path_unix_win(path)
    if subset == 'crossval':
        subset = 'train'
        true_train = False
    elif subset == 'train':
        true_train = True
    else:
        true_train = False
    csr_saver = False
    path_vocab_base = os.path.join(model_dir,
                              featuretyp, str(minTW), id0, 'vocab')
    os.makedirs(path_vocab_base, exist_ok=True)
    # hash-vectorizer is not completely stateless

    #set which version to load (that with or without 1% settings
    b =''
    if base:
        b = '_base'

    minTw_gram_auth[1].sort()
    for count, gram in enumerate(minTw_gram_auth[1]):
        set_ids[featuretyp+str(gram)] = []
        vectorizer_dic[str(gram)] ={}
        path_colids = os.path.join(path_vocab_base,
                                   '{}_grams_{}_{}_{}_authors_reduced.colids'.format(featuretyp, gram, minTW,
                                                                                     numAuthors))
        path_vocab_red = os.path.join(path_vocab_base,
                                      '{}_grams_{}_{}_{}_authors_reduced.vocab'.format(featuretyp, gram, minTW,
                                                                                       numAuthors))
        path_vocab = os.path.join(path_vocab_base,
                                  '{}_grams_{}_{}_{}_authors.vocab'.format(featuretyp, gram, minTW, numAuthors))
        if hash:
            if not vectorizers:
                vectorizer = HashingVectorizer(analyzer=ml_utils.identity_analyzer, norm=None, alternate_sign=False)
                vectorizer_dic[str(gram)]['vect'] = vectorizer
                colinds = vectorizers[str(gram)]['colinds']
            else:
                vectorizer_dic[str(gram)]['vect'] = vectorizers[str(gram)]['vect']
                vectorizer = vectorizers[str(gram)]['vect']

        if true_train:
            if not hash:
                vocab = {}

        gram_files['smpl_load'] = os.path.exists(os.path.join(*path_parts,featuretyp, str(minTW) ,id0, subset,
                                                                '{}_{}_grams_{}_{}_{}_authors.ndjson'.format(subset,
                                                                                                             featuretyp, gram, minTW, numAuthors)))

        if not gram_files['smpl_load'] and not load_sparse:
            #TODO
            sys.exit('Files not there...but they should already be there...')

        elif gram_files['smpl_load'] or load_sparse:
            #we already have a subset - simply load it
            t = os.path.join(*path_parts,featuretyp, str(minTW),id0 ,subset,
                               '{}_{}_grams_{}_{}_{}_authors.ndjson'.format(subset, featuretyp, gram, minTW, numAuthors))

            os.makedirs(os.path.join(*path_parts,featuretyp, str(minTW),id0 ,subset,'tfidf'), exist_ok= True)
            matrix_name = os.path.join(*path_parts,featuretyp, str(minTW),id0 ,subset,'tfidf',
                                       'sparse_{}_{}_grams_{}_{}_{}_authors'.format(subset, featuretyp, gram, minTW, numAuthors))

            matrix_ids = os.path.join(*path_parts, featuretyp, str(minTW), id0, subset, 'tfidf',
                                      'sparse_uID_{}_{}_grams_{}_{}_{}_authors.p'.format(subset, featuretyp, gram,
                                                                                              minTW, numAuthors))

            matrix_base = os.path.join(*path_parts, featuretyp, str(minTW), id0, subset, 'tfidf',
                                       'sparse_{}_{}_grams_{}_{}_{}_authors_base'.format(subset, featuretyp, gram, minTW,
                                                                                    numAuthors))

            sparse_exists = False
            if load_sparse:
                sparse_exists = os.path.exists(matrix_name+'.npz') and os.path.exists(matrix_ids)
                print('Sparse exists and we can load: {}'.format(sparse_exists))




        if not (load_sparse and sparse_exists):
            gram_files[str(gram)] = open(t, 'r', encoding='utf-8')



        #first we check whether the id for current line is present in ids
        #if yes, we hash - we do this for every gram type
        #set dummy in line
        lines = [1]
        #only run the loop if we do a hash or vocab does not exists

        doc_counter = 0

        if (true_train or hash) and not sparse_exists:
            id_tester = set()
            while True:
                # read the first 1000 lines
                lines = ml_utils.batch_read(gram_files[str(gram)], batchsize=1000, to_json=True)
                if not lines:
                    break
                lines = pd.DataFrame.from_dict(lines, orient='columns')
                if lines.shape[0] == 0:
                    # no ids in current lines
                    continue
                lines, id_tester = ml_utils.duplicate_uid_tester(lines, id_tester)
                if lines.shape[0] == 0:
                    # no ids in current lines
                    continue
                if id_subset:
                    lines = ml_utils.select_uids(lines, id_subset)
                    if lines.shape[0] == 0:
                        # no ids in current lines
                        continue
                if hash:
                    # save the ids in order
                    set_ids[featuretyp + str(gram)].extend(lines['uID'].to_list())
                doc_counter += lines.shape[0]
                if hash:
                    if str(gram) + '_sparse' in gram_files.keys():
                        # if we do have a sparse matrix in dict and we have to do hash
                        gram_files[str(gram) + '_sparse'] = scipy.sparse.vstack((gram_files[str(gram) + '_sparse'],
                                                                                 vectorizer.transform(
                                                                                     lines['{}_grams_{}'.format(
                                                                                         featuretyp, gram)]))
                                                                                )
                    else:
                        ##use hash vectorizer to keep in memory low
                        gram_files[str(gram) + '_sparse'] = vectorizer.transform(
                            lines['{}_grams_{}'.format(featuretyp, gram)])

                else:
                    vocab = ml_utils.update_vocab(vocab, lines, key='{}_grams_{}'.format(featuretyp, gram))

            if not hash:
                # save vocab to file
                ml_utils.make_vocab(vocab, path_vocab)
                vocab_base, _ = ml_utils.reduce_vocab(vocab, 0)
                vectorizer_base = CountVectorizer(vocabulary=vocab_base, analyzer=ml_utils.identity_analyzer)
                cutoff = 0.01
                if doc_counter >25000:
                    cutoff = 0.005
                one_percent = round(doc_counter * cutoff)
                print('Number of documents {}'.format(doc_counter))
                sys.stdout.flush()
                vocab, max_key = ml_utils.reduce_vocab(vocab, one_percent)
                # vocab, max_key = ml_utils.vocab_maker(vocab)
                # save reduced vocab to file
                ml_utils.make_vocab(vocab, path_vocab_red)

                # here we make our sparse matrix because only now do we have the full vocab
                vectorizer = CountVectorizer(vocabulary=vocab, analyzer=ml_utils.identity_analyzer)

                vectorizer_dic[str(gram)]['vect'] = vectorizer
                vectorizer_dic[str(gram)]['vocab'] = vocab


            elif hash and true_train:
                # update our csr matrix and set all words to zero with a doc-frequency below 1%
                numdocs = gram_files[str(gram) + '_sparse'].get_shape()[0]
                colsums = np.array(gram_files[str(gram) + '_sparse'].sum(axis=0)).flatten()
                prop = colsums / numdocs
                # colinds make it not stateless - we have to save this
                cutoff = 0.01
                if numdocs >25000:
                    cutoff = 0.005
                colinds = (prop > cutoff).nonzero()
                with open(path_colids, 'wb') as p:
                    pickle.dump(colinds, p)
            elif hash and not vectorizers:
                with open(path_colids, 'rb') as p:
                    colinds = pickle.load(p)
            if hash:
                gram_files[str(gram) + '_sparse_base'] = gram_files[str(gram) + '_sparse'].copy()
                gram_files[str(gram) + '_sparse'] = ml_utils.zero_columns(gram_files[str(gram) + '_sparse'], colinds)

            # important all files with their tweetids are in the same order
            # we iterate over the length of grams in ascending order...conseqently in each run we can make a tfidf-transformer/vectorizer and save it to disk
            # i.e. first for chargrams1, then for chargrams 1-2, then for chargrams1-2-3 and so forth
            # these here are the parameters taken from Custodio et al. 2021



        elif not sparse_exists:

            if vectorizers:
                vocab = vectorizer_dic[str(gram)]['vocab']
            else:
                vocab = ml_utils.get_vocab(path_vocab_red)
                # vocab = ml_utils.get_vocab(path, path_vocab)
            vocab_base = ml_utils.get_vocab(path_vocab)
            vocab_base, _  = ml_utils.reduce_vocab(vocab_base, 0)
            print('we only loaded the vocabs...')
            sys.stdout.flush()
            if not vectorizers:
                vectorizer = CountVectorizer(vocabulary=vocab, analyzer=ml_utils.identity_analyzer)
                vectorizer_dic[str(gram)] = {}
                vectorizer_dic[str(gram)]['vect'] = vectorizer
                vectorizer_dic[str(gram)]['vocab'] = vocab

            else:
                vectorizer_dic[gram] = vectorizers[str(gram)]['vect']
                vectorizer = vectorizers[str(gram)]['vect']
            vectorizer_base = CountVectorizer(vocabulary=vocab_base, analyzer=ml_utils.identity_analyzer)
        # do this for train and test
        set_ids[featuretyp + str(gram)+'_base'] = []
        set_ids[featuretyp + str(gram)] = []
        if not hash and not sparse_exists:
            if gram_files['smpl_load']:
                filelines = gram_files[str(gram)]
                filelines.seek(0)
            else:
                sys.exit('Error - wrong file to make tfidf from...')
            lines = [1]

            set_ids[featuretyp + str(gram)] = []
            while lines:
                lines = ml_utils.batch_read(filelines, batchsize=1000, to_json=True)
                if not lines:
                    break
                lines = pd.DataFrame.from_dict(lines, orient='columns')

                if id_subset:
                    lines = ml_utils.select_uids(lines, id_subset)
                    if lines.shape[0] == 0:
                        # no ids in current lines
                        continue
                # save the ids in order
                set_ids[featuretyp + str(gram)].extend(lines['uID'].to_list())
                set_ids[featuretyp + str(gram) + '_base'].extend(lines['uID'].to_list())
                # csr = ml_utils.construct_tdm(lines['{}_grams_{}'.format(featuretyp, gram)].to_list(), vocab, maxcol=max_key)
                csr = vectorizer.transform(lines['{}_grams_{}'.format(featuretyp, gram)].to_list())
                csr_base = vectorizer_base.transform(lines['{}_grams_{}'.format(featuretyp, gram)].to_list())
                # for some reason we sometimes have a coo matrix instead of a csr matrix - reform to csr if coo
                if type(csr) == scipy.sparse.coo.coo_matrix:
                    print('reform vctorizer output from coo to csr')
                    sys.stdout.flush()
                    csr = csr.tocsr()
                if type(csr_base) == scipy.sparse.coo.coo_matrix:
                    print('reform vectorizer base output from coo to csr')
                    sys.stdout.flush()
                    csr_base = csr_base.tocsr()

                if str(gram) + '_sparse' in gram_files.keys():

                    gram_files[str(gram) + '_sparse'] = scipy.sparse.vstack((gram_files[str(gram) + '_sparse'], csr))
                    gram_files[str(gram) + '_sparse_base'] = scipy.sparse.vstack((gram_files[str(gram) + '_sparse_base'], csr_base))
                else:
                    gram_files[str(gram) + '_sparse'] = csr
                    gram_files[str(gram) + '_sparse_base'] =csr_base
                # for some reason we sometimes have a coo matrix instead of a csr matrix - reform to csr if coo
                if type(gram_files[str(gram) + '_sparse']) == scipy.sparse.coo.coo_matrix:
                    print('reform from coo to csr')
                    sys.stdout.flush()
                    gram_files[str(gram) + '_sparse'] = gram_files[str(gram) + '_sparse'].tocsr()
                if type(gram_files[str(gram) + '_sparse_base']) == scipy.sparse.coo.coo_matrix:
                    print('reform base from coo to csr')
                    sys.stdout.flush()
                    gram_files[str(gram) + '_sparse_base'] = gram_files[str(gram) + '_sparse_base'].tocsr()
                #while loop ends here
                lines = [1]
            #remove doubled entries - if there are any
            set_ids[featuretyp + str(gram)], sel_ids = ml_utils.duplicate_fast_uid_tester(set_ids[featuretyp + str(gram)])
            #subset sparse array via indices
            gram_files[str(gram) + '_sparse'] = gram_files[str(gram) + '_sparse'][sel_ids,:]
            # remove doubled entries - if there are any
            set_ids[featuretyp + str(gram)+'_base'], sel_ids = ml_utils.duplicate_fast_uid_tester(set_ids[featuretyp + str(gram)+'_base'])
            #subset sparse array via indices
            gram_files[str(gram) + '_sparse'+'_base'] = gram_files[str(gram) + '_sparse'+'_base'][sel_ids, :]

            print('Have {} (documents,features) in in matrix...'.format(gram_files[str(gram) + '_sparse'].shape))
            sys.stdout.flush()



        if not sparse_exists:
            ml_utils.save_sparse(gram_files[str(gram) + '_sparse'], path=matrix_name)
            ml_utils.save_uids(set_ids[featuretyp + str(gram)], path=matrix_ids)

            ml_utils.save_sparse(gram_files[str(gram) + '_sparse_base'], path=matrix_base)
        else:
            gram_files[str(gram) + '_sparse'] = ml_utils.load_sparse(matrix_name)
            set_ids[featuretyp + str(gram)] = ml_utils.load_uids(matrix_ids)

            gram_files[str(gram) + '_sparse_base'] = ml_utils.load_sparse(matrix_base)

            #load dictionary
            vocab = ml_utils.get_vocab(path_vocab_red)
            vectorizer = CountVectorizer(vocabulary=vocab, analyzer=ml_utils.identity_analyzer)
            vectorizer_dic[str(gram)] = {}
            vectorizer_dic[str(gram)]['vect'] = vectorizer
            vectorizer_dic[str(gram)]['vocab'] = vocab
        # add csr column-wise together

        if type(gram_files[str(gram) + '_sparse']) == scipy.sparse.coo.coo_matrix:
            print('reform from coo to csr')
            sys.stdout.flush()
            gram_files[str(gram) + '_sparse'] = gram_files[str(gram) + '_sparse'].tocsr()
        if type(gram_files[str(gram) + '_sparse_base']) == scipy.sparse.coo.coo_matrix:
            print('reform base from coo to csr')
            sys.stdout.flush()
            gram_files[str(gram) + '_sparse_base'] = gram_files[str(gram) + '_sparse_base'].tocsr()

            # add csr column-wise together
        if csr_saver:
            vectorizer_dic[str(gram)]['vect'] = 0
            # make sure matrices have same dimensions

            fin_csr, \
            set_ids[featuretyp + str(minTw_gram_auth[1][count-1])], \
            gram_files[str(gram)+ '_sparse' +b], set_ids[featuretyp + str(gram)] = ml_utils.assess_equal(fin_csr,
                                                                                              set_ids[featuretyp + str(minTw_gram_auth[1][count-1])],
                                                                                              gram_files[str(gram) + '_sparse'+b],
                                                                                              set_ids[featuretyp + str(gram)])
            try:
                assert fin_csr.shape[0] == gram_files[str(gram) + '_sparse' +b].shape[0]
            except AssertionError as e:
                e.args += ('feature: {}, minTw: {}, grams:{}, numAuthors:{}'.format(featuretyp, minTW, gram ,numAuthors))
                raise
            vectorizer_dic[str(gram)]['feature_inds'] = gram_files[str(gram) + '_sparse' + b].shape
            fin_csr = scipy.sparse.hstack((fin_csr, gram_files[str(gram) + '_sparse'+b]))
        else:
            vectorizer_dic[str(gram)]['vect'] = 0
            vectorizer_dic[str(gram)]['feature_inds'] = gram_files[str(gram) + '_sparse' + b].shape
            csr_saver = True
            fin_csr = gram_files[str(gram) + '_sparse'+b].copy()

            #close files
            if not load_sparse:
                gram_files[str(gram)].close()
            del gram_files[str(gram) + '_sparse'+b]
            gc.collect()

        #now we make our tfidf transformation
        gramsstring = '_'.join([str(el) for el in minTw_gram_auth[1] if el <= gram])
        if not trans_in:
            if true_train:
                trans = TfidfTransformer(norm ='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
                trans = trans.fit(fin_csr)

                os.makedirs(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]),id0), exist_ok=True)
                with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                                       'tfidf_{}_grams_{}_{}_{}_hash_{}{}.p'.format(featuretyp, gramsstring, minTW, numAuthors ,hash,b)), 'wb') as p:
                    pickle.dump(trans, p)
            else:
                with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                                       'tfidf_{}_grams_{}_{}_{}_hash_{}{}.p'.format(featuretyp, gramsstring, minTW, numAuthors ,hash,b)), 'rb') as p:
                    trans = pickle.load(p)
    if trans_in:
        trans = trans_in

    tfidf = trans.transform(fin_csr)
    trans = 0
    return trans, vectorizer_dic, tfidf, set_ids

def make_scaler(featuretyp:str, minTw_gram_auth:list, path:str, id_type:str, subset ='train', id_subset=[], scaler =None, load_scaled=True):
    # function to scale numerical features between 0 and 1
    set_ids = {}
    set_ids[featuretyp+''] = []

    if featuretyp == 'num':
        keys = ["init_len", "prepr_len", "mentions", "tags", "urls", "times",
                "emotic_num", 'emojis_num', 'numericals']
    else:
        keys = ['polarity', 'subjectivity']


    if subset == 'crossval':
        subset = 'train'
        true_train = False
    elif subset == 'train':
        true_train = True
    else:
        true_train = False
    path_parts = ml_utils.split_path_unix_win(path)
    id0 = id_type.split('_')[0]
    model_dir = ml_utils.make_save_dirs(path)
    subset_path = os.path.join(*path_parts, featuretyp, str(minTw_gram_auth[0]), id0, subset)
    os.makedirs(os.path.join(subset_path, 'scaled'), exist_ok=True)
    #read in file
    scaled_path = os.path.join(subset_path, 'scaled', "scaled_{}_{}_{}_{}_authors".format(subset,featuretyp,
                                                                                                 minTw_gram_auth[0],
                                                                                                 minTw_gram_auth[2]))
    ids_path = os.path.join(subset_path, 'scaled',"scaled_uID_{}_{}_{}_{}_authors.p".format(subset,
                                                                                                    featuretyp,
                                                                                                    minTw_gram_auth[0],
                                                                                                    minTw_gram_auth[2]))
    scaled_exists = os.path.exists(scaled_path+'.npy') & os.path.exists(ids_path)
    if not load_scaled or not scaled_exists:
        with open(os.path.join(subset_path, "{}_{}_{}_{}_authors.ndjson".format(subset,featuretyp,minTw_gram_auth[0],minTw_gram_auth[2])),
                  'r', encoding='utf-8') as f:
            dat = ndjson.load(f)

        dat = pd.DataFrame.from_dict(dat, orient='columns')
        dat.drop_duplicates(subset=['uID'], keep='first', inplace=True)
        dat.reset_index(drop=True, inplace=True)
        print('Shape after dropping doubled uIDs: {}'.format(dat.shape))
        for key in keys:
            if dat[key].dtype != 'float':
                print('scale {}'.format(key))
                sys.stdout.flush()
                dat[key] = dat[key].apply(lambda x: statistics.mean(x))

        if id_subset:
            dat = ml_utils.select_uids(dat, id_subset)

        if true_train:
            scaler = StandardScaler(copy=False)
            # save the ids in order
            set_ids[featuretyp+''].extend(dat['uID'].to_list())
            print('fit scaler')
            sys.stdout.flush()
            dat = scaler.fit_transform(dat[keys].to_numpy())
            print('fitted scaler...')
            sys.stdout.flush()

            #save scaler
            with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[0], minTw_gram_auth[2])), 'wb') as p:
                pickle.dump(scaler, p)
        else:
            if not scaler:
                with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[0], minTw_gram_auth[2])), 'rb') as p:
                    scaler = pickle.load( p)


            # save the ids in order
            set_ids[featuretyp+''].extend(dat['uID'].to_list())
            dat = scaler.transform(dat[keys].to_numpy())


        if not id_subset:
            np.save(file=scaled_path, arr=dat)
            with open(os.path.join(subset_path, 'scaled',"scaled_uID_{}_{}_{}_{}_authors.p".format(subset,
                                                                                                            featuretyp,
                                                                                                            minTw_gram_auth[0],
                                                                                                            minTw_gram_auth[2])),'wb') as f:
                pickle.dump(set_ids, f)

    else:

        dat = np.load(file=scaled_path+'.npy')
        with open(os.path.join(subset_path, 'scaled',"scaled_uID_{}_{}_{}_{}_authors.p".format(subset,
                                                                                                        featuretyp,
                                                                                                        minTw_gram_auth[0],
                                                                                                        minTw_gram_auth[2])),'rb') as f:
            set_ids = pickle.load(f)

        if not scaler:
            with open(os.path.join(model_dir, featuretyp, str(minTw_gram_auth[0]), id0,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[0], minTw_gram_auth[2])), 'rb') as p:
                scaler = pickle.load(p)

    scaler = 0
    return scaler, dat, set_ids

def process_input(featuretyp:str, target:str, minTw_gram_auth, path:str, subset='train', id_subset=[], components=None, hash=False, skip_pca=True, base=False):
    # processes the different kinds of feature inputs (character-based, word-based, pos-tag based etc.) and applies the respective transformation-routines
    #find the number of features for pca which explains 99% variance - this takes a while
    id_type = target + '_ids'

    #depending on type of data we have to vectorize the data first
    if not components:
        components = {}
    if featuretyp in ['num', 'polarity']:
        components['vect_type'] = components.get('vect_type', 'scaler')
        scaler = components.get('vectorizing', None)
        if scaler:
            scaler =scaler[0]
        scaler, dat, ids = make_scaler(featuretyp, minTw_gram_auth, path, id_type, subset = subset, id_subset=id_subset, scaler=scaler,
                                       load_scaled=False)
        components.update({'vectorizing': [scaler]})

    elif featuretyp == 'emoticon_c':
        components['vect_type'] = components.get('vect_type', 'vect_scaler')
        scaler = components.get('vectorizing', [None, None])
        if scaler:
            scaler =scaler[0]
        vectorizer, dat, ids =  make_emoji_vector(featuretyp, minTw_gram_auth, path, id_type, subset = subset, id_subset=id_subset,
                                                  vectorizer=scaler, load_sparse=False)

        components.update({'vectorizing': vectorizer})

    else:
        components['vect_type'] = components.get('vect_type', 'tfidf_vect')
        vectorizers = components.get('vectorizing', [None, None])
        vectorizer_dic = vectorizers[0]
        transformer = vectorizers[1]


        tfidf_trans, vectorizer_dic, dat, ids = make_tfidf(featuretyp, minTw_gram_auth, path, id_type, subset=subset,
                                                    id_subset=id_subset, vectorizers=vectorizer_dic, trans_in=transformer, hash=hash, load_sparse=True, base=base)

        components.update({'vectorizing': [vectorizer_dic, tfidf_trans]})



    return components, dat, ids

def encode_y(y, target, model_dir, filename, id_subset, subset='train'):
    # function to encode the y-values/target
    if subset == 'crossval':
        subset = 'train'
        true_train = False
    elif subset == 'train':
        true_train = True
    else:
        true_train = False
    cats, target_key = ml_utils.y_categories_target_key(target)
    #encoder = OneHotEncoder(drop='if_binary', categories=[cats])
    encoder = OneHotEncoder(categories=[cats])
    if true_train or not os.path.exists(os.path.join(model_dir, filename)):
        y_encoded = encoder.fit_transform(y[target_key].to_numpy().reshape((y.shape[0], -1)))
        with open(os.path.join(model_dir, filename), 'wb') as w:
            pickle.dump(encoder, w)
    else:
        with open(os.path.join(model_dir, filename), 'rb') as w:
            encoder = pickle.load(w)
        y_encoded = encoder.transform(y[target_key].to_numpy().reshape((y.shape[0], -1)))


    return np.array(y_encoded.todense()), cats, target_key

def prepare_data(X_train, y_train, trainids, encoder_name, model_dir, target,
                 X_test=None, y_test=None, testids=None, subset='train'):

    '''assesses the order of the different datasets, i.e. feature-input and
    target set as they may be constructed at different points in time or with interruptions'''

    y_train_ids = y_train.uID.to_numpy().flatten()
    y_test_ids = y_test.uID.to_numpy().flatten()


    X_train, trainids = ml_utils.assess_order(sorter=y_train_ids, data=X_train, data_ids=np.array(trainids).flatten(),
                                              return_ids=True)
    if type(X_test) != type(None):
        X_test, testids = ml_utils.assess_order(sorter=y_test_ids, data=X_test, data_ids=np.array(testids).flatten(),
                                                return_ids=True)
    y_train_enc, cats, target_key = encode_y(y_train, target, model_dir, encoder_name, id_subset=[], subset=subset)

    if type(X_test)  == type(None):
        return X_train, trainids, y_train_enc, y_train_ids, cats, target_key
    else:
        return X_train, trainids, y_train_enc, y_train_ids, cats, target_key, X_test, testids, y_test_ids

def _direct_predict(X_train:scipy.sparse.csr.csr_matrix, trainids:list, X_test:scipy.sparse.csr.csr_matrix,
                    testids:list,y_train:pd.DataFrame, y_test:pd.DataFrame,info:list, path:str, rewrite=False, q=None):

    # function for direct prediction of target using multiple feature types as input (i.e. combination of character-based, word-based etc. features)

    #info=[target, tweetLen, numAuth, '_'.join(featuretypes), subgrams]
    path = os.path.join(*ml_utils.split_path_unix_win(path))
    ml_utils.make_save_dirs(path)
    model_dir = os.path.join(path, 'direct_model', str(info[1]), info[0])
    model_dir = ml_utils.make_save_dirs(model_dir)
    pred_path = ml_utils.make_prediction_dir(path=path, minTweet=info[1], target=info[0], featuretype=info[3])
    os.makedirs(pred_path, exist_ok=True)
    encoder_name = 'encoder_{}_{}_{}_authors.p'.format(info[0], info[1], info[2])
    mode = 'a'
    if rewrite:
        mode = 'w'
    if type(q) != type(None):
        mode = 'return'
    gramstring = '_'.join([str(gram) for gram in info[4]])
    if not rewrite:
        weight_path = os.path.join(pred_path,
                               'direct_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3], gramstring,
                                                                                     info[1], info[2]))
        if os.path.exists(weight_path):
            print('load weight dic for direct...')
            sys.stdout.flush()
            with open(os.path.join(pred_path,
                                   'direct_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3], gramstring,
                                                                                         info[1], info[2])), 'rb') as w:
                weight_dic = pickle.load(w)
            with open(os.path.join(pred_path,
                                   "direct_pred_{}_{}_{}_{}_{}_{}.json".format('test', info[0], info[3], gramstring,
                                                                              info[1], info[2])),
                      'r') as w:
                y_test = pd.read_json(w)
            cats, target_key = ml_utils.y_categories_target_key(info[0])
            ret = evaluations.make_score_table(info=info, y_true=y_test[target_key].to_numpy(),
                                             y_pred=y_test[info[0]+ '_predicted_cat'].to_numpy(), path=path, modeltype='direct', mode=mode)
            if type(q) != type(None):
                assert type(ret) == dict
                q.put(ret)

            return weight_dic
    #encode y
    X_train, trainids, y_train_enc, y_train_ids,\
    cats, target_key, X_test, testids, y_test_ids = prepare_data(X_train, y_train, trainids, encoder_name, model_dir, info[0],
                 X_test=X_test, y_test=y_test, testids=testids, subset='val')


    print('train direct...')
    sys.stdout.flush()
    svm = SGDClassifier(loss='hinge', penalty='l2')

    svm = ml_utils.train_scikit_model(model=svm, X_train=X_train, y_train_enc=y_train_enc, cats=cats, batch_size=1000)

    print('make predictions...')
    sys.stdout.flush()
    preds, preds_cat = ml_utils.make_predictions(model=svm, X_test=X_test, cats=cats, batch_size=1000)

    with open(os.path.join(model_dir, 'direct_model_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3],gramstring, info[1], info[2])),
              'wb') as w:
        pickle.dump(svm, w)


    print('saving predictions...')
    y_test[info[0] + '_predicted_cat'] = preds_cat
    with open(os.path.join(pred_path, "direct_pred_{}_{}_{}_{}_{}_{}.json".format('test', info[0], info[3], gramstring,info[1], info[2])),
              'w') as w:
        y_test.to_json(w)

    ret = evaluations.make_score_table(info=info, y_true=y_test[target_key].to_numpy(), y_pred=np.array(preds_cat), path=path, modeltype='direct', mode=mode)
    if type(q) != type(None):
        assert type(ret) == dict
        q.put(ret)
    weight_dic = evaluations.calc_weight_dic(weights=svm.coef_, info=info, modeltype='direct')
    with open(os.path.join(pred_path,
                           'direct_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3], gramstring,
                                                                                  info[1], info[2])), 'wb') as w:
        pickle.dump(weight_dic, w)
    print('finished with direct predict...')
    sys.stdout.flush()
    return weight_dic

def _dynAA(X_train:np.ndarray, trainids:list, X_test:np.ndarray, testids:list,
           y_train:pd.DataFrame, y_test:pd.DataFrame,info:list, path:str, weights:list, rewrite = True, q=None):

    '''implements a version of the dynAA model by Custodio et al. 2021,
    i.e. stacked classifier with first-layer classifiers using only one type
    of feature and second layer predicting based on the first'''

    path = os.path.join(*ml_utils.split_path_unix_win(path))
    ml_utils.make_save_dirs(path)
    model_dir = os.path.join(path, 'dynAA', str(info[1]), info[0])
    model_dir = ml_utils.make_save_dirs(model_dir)
    pred_path = ml_utils.make_prediction_dir(path=path, minTweet=info[1], target=info[0], featuretype=info[3])
    os.makedirs(pred_path, exist_ok=True)
    encoder_name = 'encoder_{}_{}_{}_authors.p'.format(info[0], info[1], info[2])
    mode = 'a'
    if rewrite:
        mode = 'w'
    if type(q) != type(None):
        mode = 'return'
    gramstring = '_'.join([str(gram) for gram in info[4]])
    if not rewrite:
        weight_path = os.path.join(pred_path,
                               'dynAA_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3], gramstring,
                                                                                     info[1], info[2]))
        if os.path.exists(weight_path):
            print('load weight dic for dynAA...')
            sys.stdout.flush()
            with open(os.path.join(pred_path,
                                   'dynAA_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3], gramstring,
                                                                                         info[1], info[2])), 'rb') as w:


                weight_dic = pickle.load(w)

            with open(os.path.join(pred_path,
                                   "dynAA_pred_{}_{}_{}_{}_{}_{}.json".format('test', info[0], info[3], gramstring,
                                                                              info[1], info[2])),
                      'r') as w:
                y_test = pd.read_json(w)
            cats, target_key = ml_utils.y_categories_target_key(info[0])
            ret = evaluations.make_score_table(info=info, y_true=y_test[target_key].to_numpy(),
                                             y_pred=y_test[info[0]+ '_predicted_cat'].to_numpy(), path=path, modeltype='dynAA', mode=mode)
            if type(q) != type(None):
                assert type(ret) == dict
                q.put(ret)

            return weight_dic

    #encode y
    #encode
    X_train, trainids, y_train_enc, y_train_ids,\
    cats, target_key, X_test, testids, y_test_ids = prepare_data(X_train, y_train, trainids, encoder_name, model_dir, info[0],
                 X_test=X_test, y_test=y_test, testids=testids, subset='val')



    #make model
    model = SGDClassifier(loss='log', penalty='l2', random_state=123456, verbose=0)
    print('train DynAA...')
    sys.stdout.flush()
    model = ml_utils.train_scikit_model(model, X_train=X_train, y_train_enc=y_train_enc, cats=cats,
                                        batch_size=1000, epochs=50)
    gramstring = '_'.join([str(gram) for gram in info[4]])
    with open(os.path.join(model_dir, 'dynAA_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3],gramstring, info[1], info[2])), 'wb') as w:
        pickle.dump(model, w)

    preds ,preds_cat = ml_utils.make_predictions(model, X_test=X_test, cats=cats)
    print('saving predictions...')
    y_test[info[0] + '_predicted_cat'] = preds_cat
    with open(os.path.join(pred_path, "dynAA_pred_{}_{}_{}_{}_{}_{}.json".format('test', info[0], info[3], gramstring,info[1], info[2])),
              'w') as w:
        y_test.to_json(w)

    ret = evaluations.make_score_table(info=info, y_true=y_test[target_key].to_numpy(), y_pred=np.array(preds_cat), path=path, modeltype='dynAA', mode=mode)
    if type(q) != type(None):
        assert type(ret) == dict
        #then ret is not None
        q.put(ret)

    weight_dic = evaluations.calc_updated_weights(weights, model.coef_, info, modeltype = 'dynAA')
    #slice the weights to list again for compatibility


    with open(os.path.join(pred_path, 'dynAA_weights_{}_{}_grams_{}_{}_{}_authors.p'.format(info[0], info[3],gramstring, info[1], info[2])), 'wb') as w:
        pickle.dump(weight_dic, w)


    print('finished with DynAA...')
    sys.stdout.flush()

    return weight_dic

def load_train(featuretyp, target, minTw_gram_auth:list, path:str, y:pd.DataFrame, hash:bool, subset ='train', true_train=True):
    # simply load if already trained
    components = None
    if true_train:
        print('we are truly fitting and predicting now...')
        sys.stdout.flush()
        if subset == 'train':
            components, datainfo = train_comp(featuretyp, target, minTw_gram_auth=minTw_gram_auth,
                                                    path=path, y=y, hash=hash, id_subset=[])
        else:
            datainfo = evaluate_comp(featuretyp, target, minTw_gram_auth=minTw_gram_auth, id_subset=[],
                                            subset=subset, path=path, y=y, hash=hash)
    else:
        print('try fetching and loading...')
        sys.stdout.flush()
        pred_path = ml_utils.make_prediction_dir(path, featuretyp, minTw_gram_auth[0], target)
        exists = False
        gramstring = '_'.join([str(el) for el in minTw_gram_auth[1]])
        if os.path.exists(os.path.join(pred_path, "datainfo_{}_{}_{}_{}_{}_{}.p".format(subset, target, featuretyp, gramstring,
                                                                                minTw_gram_auth[0],
                                                                                minTw_gram_auth[2]))):
            exists = True

        if not exists:
            print('we are have to fit or predict after all...')
            sys.stdout.flush()
            if subset == 'train':
                components, datainfo = train_comp(featuretyp, target, minTw_gram_auth=minTw_gram_auth,
                                                        path=path, y=y, id_subset=[], hash=hash)
            else:
                datainfo = evaluate_comp(featuretyp, target, minTw_gram_auth=minTw_gram_auth,
                                                         subset=subset, path=path, y=y, id_subset=[], hash=hash)

        else:
            with open(os.path.join(pred_path, "datainfo_{}_{}_{}_{}_{}_{}.p".format(subset, target, featuretyp, gramstring,
                                                                                    minTw_gram_auth[0],
                                                                                    minTw_gram_auth[2])), 'rb') as w:


                print('Warning: Loading no components...')
                sys.stdout.flush()
                datainfo = pickle.load(w)

                #check whether data is in dict or we have to load it separately
                if type(datainfo['data']) == type('string'):
                    if '.npz' in datainfo['data']:
                        datainfo['data'] = ml_utils.load_sparse(datainfo['data'])
                    elif '.npy' in datainfo['data']:
                        datainfo['data'] = np.load(datainfo['data'])
                    elif '.p' in datainfo['data']:
                        with open(datainfo['data'], 'rb') as r:
                            datainfo['data'] = pickle.load(r)

    return components, datainfo

def train_loop(path: str, in_dic:dict, targets:str, dynAA = True, partial =False, singular=False, hash=False,
               true_train=True, baseline_fetch=True, all_grams=True, q=None):
    #loop for training of a model on each of the different subsets of feature types - multiprocessing so each ChildDaemon trains by themselves
    pid = mp.current_process().pid
    #function iterates over files and makes all classifiers etc as predictions and saves them to file
    mode = 'a'
    #if true_train:
    #    mode = 'w'
    if type(q) != type(None):
        mode2 = 'return'
        print('set mode to return...')
        sys.stdout.flush()
    else:
        mode2 = mode
    #gram_dict = ml_utils.make_gram_dict(path)
    #gram_dict = ml_utils.reformat_gram_dic(gram_dict)
    order = ['dist', 'char', 'asis', 'pos', 'tag', 'dep', 'lemma', 'word', 'emoticon_c', 'polarity', 'num']
    re_test = re.compile('train')
    #either make the predictions for pseudo-dynAA or make dic in correct form for overall data in one model
    subset = 'train'
    targets = np.array([targets]).flatten().tolist()
    #variables for saving across featuretypes
    t_dat = None
    t_dat_ids = None
    tst_dat_ids = None
    v_pred = None
    v_y_ids = None
    tst_y_ids = None
    tst_pred = None
    tst_dat = None
    t_weights = None
    t_vocab = None
    t_inds = None
    tst_shape = None
    weights_dic = {}
    baseline_fetch = not baseline_fetch
    #fix path
    feature_keys = sorted(list(in_dic.keys()), key=order.index)
    path = os.path.join(*ml_utils.split_path_unix_win(path))
    for target in targets:
        for tweetLen in in_dic[feature_keys[0]][0]:
            for numAuth in in_dic[feature_keys[0]][2]:
                featuretypes = {'types':[], 'grams':[]}
                for featuretyp in feature_keys:
                    featuretypes['types'].append(featuretyp)
                    grams = []
                    print('doing featuretype {}'.format(featuretyp))

                    for gram in in_dic[featuretyp][1]:
                        grams.append(gram)

                    grams.sort()
                    y_path = os.path.join('{}_ids'.format(target), subset,'{}_ids_{}_{}_{}_authors_balanced.json'.format(subset, target, tweetLen, numAuth))
                    featuretypes['grams'].append(grams)
                    y_val = re.sub(re_test, 'val', y_path)
                    y_test = re.sub(re_test, 'test', y_path)
                    y = pd.read_json(os.path.join(path, y_path))
                    y_val = pd.read_json(os.path.join(path, y_val))
                    y_test = pd.read_json(os.path.join(path, y_test))
                    #iterate over all grams, i.e. make results for char-1, char-1-2, char-1-2-3 etc.
                    for i in range(1, len(grams)+1):
                        subgrams = copy.deepcopy(grams[0:i])
                        if not all_grams:
                            #if this is true we only do the specific block of grams we have gotten and break the loop afterwards...(memory intense)
                            subgrams = copy.deepcopy(grams)

                        print('{} is doing {} with tw {} and authors {} for feature {} with grams {}'.format(pid, target, tweetLen,
                                                                                                             numAuth, featuretypes['types'], subgrams))
                        sys.stdout.flush()
                        _, datainfo_train = load_train(featuretyp, target, minTw_gram_auth=[tweetLen, subgrams, numAuth],
                                                           path=path, y = y, hash=hash, subset='train', true_train=baseline_fetch)

                        _, datainfo_val = load_train(featuretyp, target, minTw_gram_auth=[tweetLen, subgrams, numAuth],
                                                            subset ='val', path=path, y = y_val, hash=hash, true_train=baseline_fetch)
                        # results also on test_set for comparability
                        _, datainfo_test = load_train(featuretyp, target,
                                                             minTw_gram_auth=[tweetLen, subgrams, numAuth],
                                                             subset='test', path=path, y=y_test, hash=hash, true_train=baseline_fetch)

                        del _


                        if in_dic[feature_keys[0]][2].index(numAuth) != 0 and (order.index(featuretyp) == 0 or singular) and not partial:

                            print('\n Comparing the results from {} auhtors to {} authors for feature {} and grams {}'.format(
                                weights_dic[featuretyp + str(subgrams[-1])]['numAuth'], numAuth, featuretyp, subgrams[-1]))
                            #we are after first authorNum iteration and only do this for the first featuretype - otherwise direct model func will do this
                            weights = [weights_dic[featuretyp+str(subgrams[-1])]['weights'], datainfo_train['weights']] #prev and current
                            slice_inds = [[weights_dic[featuretyp + str(subgrams[-1])]['slice_inds']], [datainfo_train['slice_inds']]] #prev and current
                            if featuretyp not in ['emoticon_c''polarity', 'num']:
                                vocabs = [weights_dic[featuretyp + str(subgrams[-1])]['vocabs'], datainfo_train['vocabs']] #previous vocab and current
                                #resort weight matrices, so that features at same position are the same features...
                                weights = ml_utils.sort_weight_matrices(weights, vocabs)
                            ret = evaluations.make_distortion_table(lower_dic={'modeltype':'direct', 'weights':weights[0]},
                                                              upper_dic={'modeltype':'direct', 'weights':weights[1]},
                                                              slice_inds = slice_inds,
                                                              info=[target, tweetLen,numAuth,'_'.join(featuretypes['types']), subgrams],
                                                              modeltype='direct', mode=mode2, path=path)
                            if mode2 == 'return':
                                assert type(ret) == dict
                                q.put(ret)

                        if order.index(featuretyp) == 0 or singular:
                            # save new weights in weight dic for next round of numAuth
                            weights_dic[featuretyp + str(subgrams[-1])] = {}
                            weights_dic[featuretyp + str(subgrams[-1])]['weights'] = datainfo_train['weights']
                            weights_dic[featuretyp + str(subgrams[-1])]['slice_inds'] = datainfo_train['slice_inds']
                            weights_dic[featuretyp + str(subgrams[-1])]['vocabs'] = datainfo_train['vocabs'] #vocab to sort weight matrices with before slicing
                            weights_dic[featuretyp + str(subgrams[-1])]['numAuth'] =numAuth

                        if type(datainfo_train['data']) != scipy.sparse.csr.csr_matrix:
                            # convert our numpy array to sparse matrix
                            datainfo_train['data'] = scipy.sparse.csr_matrix(datainfo_train['data'])
                        if type(datainfo_val['data']) != scipy.sparse.csr.csr_matrix:
                                # convert our numpy array to sparse matrix
                                datainfo_val['data'] = scipy.sparse.csr_matrix(datainfo_val['data'])
                        if type(datainfo_test['data']) != scipy.sparse.csr.csr_matrix:
                                # convert our numpy array to sparse matrix
                                datainfo_test['data'] = scipy.sparse.csr_matrix(datainfo_test['data'])


                        if len(featuretypes['types'])>1:

                            #can only be in the second featuretyp-interation onwards
                            print('Doing featuretypes {} for dynAA and/or direct'.format(featuretypes['types']))
                            sys.stdout.flush()
                            if len(datainfo_val['predictions'].shape) == 1:
                                datainfo_val['predictions'] = datainfo_val['predictions'].reshape((-1, 1))
                                datainfo_test['predictions'] = datainfo_test['predictions'].reshape((-1, 1))
                            # make sure that we have the same ids and the same order of them everywhere (should be by default but meh)
                            #for dynAA
                            tmp_y_pred, tmp_y_ids, datainfo_val['predictions'], datainfo_val['y_ids'] = ml_utils.assess_equal(v_pred, np.array(v_y_ids[-1]),
                                                                                                                 datainfo_val['predictions'],
                                                                                                                 np.array(datainfo_val['y_ids']))

                            tmp_y_pred_test, tmp_y_ids_test,\
                            datainfo_test['predictions'], datainfo_test['y_ids'] = ml_utils.assess_equal(tst_pred, np.array(tst_y_ids[-1]),
                                                                                                         datainfo_test['predictions'],
                                                                                                         np.array(datainfo_test['y_ids']))


                            tmp, tmp_ids, datainfo_train['data'], datainfo_train['data_ids'] = ml_utils.assess_equal(t_dat,
                                                                                                                     np.array(t_dat_ids[-1]),
                                                                                                                 scipy.sparse.vstack([datainfo_train['data'],
                                                                                                                                  datainfo_val['data']],
                                                                                                                                  format='csr'),
                                                                                                                 np.array(list(datainfo_train['data_ids'])+
                                                                                                                          list(datainfo_val['data_ids'])))
                            tst_tmp, tst_tmp_ids, datainfo_test['data'], datainfo_test['data_ids'] = ml_utils.assess_equal(tst_dat,np.array(tst_dat_ids[-1]),
                                                                                                                           datainfo_test['data'],
                                                                                                                           datainfo_test['data_ids'])


                            #hstack for direct predictions
                            tmp = scipy.sparse.hstack([tmp,datainfo_train['data']], format='csr')
                            tst_tmp = scipy.sparse.hstack([tst_tmp,datainfo_test['data']], format='csr')

                            #hstack also the y_val predictions for dynAA
                            tmp_y_pred = np.hstack([tmp_y_pred, datainfo_val['predictions']])
                            tmp_y_pred_test = np.hstack([tmp_y_pred_test, datainfo_test['predictions']])

                            #make tmp for weights
                            tmp_weights = t_weights +[datainfo_train['weights']]
                            assert len(tmp_weights) == len(featuretypes['types']) #test wether we have at max so many feat-types as weightsmats
                            tmp_slice_inds = t_inds + [datainfo_train['slice_inds']]
                            if len(datainfo_train['vocabs']) >0:
                                tmp_vocab = t_vocab + [datainfo_train['vocabs']]

                            if len(featuretypes['types'])>1 and not partial or partial and len(featuretypes['types']) == len(feature_keys):
                                if dynAA or dynAA == 'both':
                                    print('Doing DynAA...')
                                    sys.stdout.flush()
                                    weights_dynAA = _dynAA(X_train=tmp_y_pred, trainids=tmp_y_ids, X_test= tmp_y_pred_test, testids = tmp_y_ids_test,
                                            y_train=y_val, y_test=y_test, info=[target, tweetLen,numAuth, '_'.join(featuretypes['types']), subgrams], path=path,
                                            weights = tmp_weights, rewrite=true_train, q=q)
                                    #make distortion if we are in loop after first, otherwise just save relevance
                                    if 'dynAA'+featuretyp + str(subgrams[-1]) in weights_dic.keys():
                                        print('I am in here for distortion dynAA')
                                        sys.stdout.flush()
                                        prev_weights = weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['weights'] #this is weight matrix
                                        prev_inds = weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['slice_inds']
                                        prev_vocabs = weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['vocabs']
                                        if len(tmp_vocab) > 0:  # check whether we have stacked features whichs' order varies
                                            prev_weights, curr_weights = ml_utils.sort_direct_classifier_weights(weights1=prev_weights['weights'],
                                                                                                                 weights2=weights_dynAA['weights'],
                                                                                                                 vocabs1=prev_vocabs,
                                                                                                                 vocabs2=tmp_vocab,
                                                                                                                 inds1=prev_inds,
                                                                                                                 inds2=tmp_slice_inds,
                                                                                                                 featuretypes=featuretypes['types'])
                                        curr_dic = copy.deepcopy(weights_dynAA)
                                        prev_dic = weights_dic.pop('dynAA' + featuretyp + str(subgrams[-1])) # so that we do not grow forever
                                        curr_dic['weights']= curr_weights
                                        prev_dic['weights']['weights'] = prev_weights


                                        #make distortion
                                        ret = evaluations.make_distortion_table(
                                            lower_dic=prev_dic['weights'],
                                            upper_dic=curr_dic,
                                            slice_inds=[prev_inds, tmp_slice_inds],
                                            info=[target, tweetLen, numAuth, '_'.join(featuretypes['types']), subgrams],
                                            modeltype='dynAA', mode=mode2, path=path)

                                        if mode2 == 'return':
                                            assert type(ret) == dict
                                            sys.stdout.flush()
                                            q.put(ret)
                                    weights_dic['dynAA' + featuretyp + str(subgrams[-1])] = {}
                                    weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['weights'] = weights_dynAA #this here is a dictionary
                                    weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['slice_inds'] = tmp_slice_inds  # same inds for all three
                                    weights_dic['dynAA' + featuretyp + str(subgrams[-1])]['vocabs'] = tmp_vocab

                                # # #do direct logistic with large column matrix
                                if not dynAA or dynAA == 'both':
                                    print('Doing direct predict...')
                                    sys.stdout.flush()
                                    weights_direct = _direct_predict(X_train=tmp, trainids=tmp_ids, X_test=tst_tmp,
                                                                     testids = tst_tmp_ids, y_train = y, y_test=y_test,
                                                                     info=[target, tweetLen,numAuth, '_'.join(featuretypes['types']), subgrams],
                                                                     path=path, rewrite=true_train, q=q)
                                    if 'direct'+featuretyp + str(subgrams[-1]) in weights_dic.keys():
                                        print('I am in here for distortion direct')
                                        sys.stdout.flush()
                                        ##resort weights in the same order

                                        prev_weights = weights_dic['direct' + featuretyp + str(subgrams[-1])]['weights']
                                        prev_inds = weights_dic['direct' + featuretyp + str(subgrams[-1])]['slice_inds']
                                        prev_vocabs = weights_dic['direct' + featuretyp + str(subgrams[-1])]['vocabs']
                                        #resort weights in correct order for previous
                                        if len(tmp_vocab)>0: #check whether we have stacked features whichs order varies
                                            prev_weights, curr_weights = ml_utils.sort_direct_classifier_weights(weights1=prev_weights['weights'],
                                                                                weights2=weights_direct['weights'],
                                                                                vocabs1=prev_vocabs,
                                                                                vocabs2=tmp_vocab,
                                                                                inds1 = prev_inds,
                                                                                inds2=tmp_slice_inds,
                                                                                featuretypes=featuretypes['types'])

                                        curr_dic = copy.deepcopy(weights_direct)
                                        prev_dic = weights_dic.pop('direct'+featuretyp + str(subgrams[-1]))
                                        curr_dic['weights'] = curr_weights
                                        prev_dic['weights']['weights'] = prev_weights
                                        #make distortion
                                        ret = evaluations.make_distortion_table(
                                            lower_dic=prev_dic['weights'],
                                            upper_dic=curr_dic,
                                            slice_inds=[prev_inds, tmp_slice_inds],
                                            info=[target, tweetLen, numAuth, '_'.join(featuretypes['types']), subgrams],
                                            modeltype='direct', mode=mode2, path=path)
                                        if mode2 == 'return':
                                            assert type(ret) == dict
                                            q.put(ret)
                                    weights_dic['direct' + featuretyp + str(subgrams[-1])] = {}
                                    weights_dic['direct' + featuretyp + str(subgrams[-1])]['weights'] = weights_direct #this here is a dictionary
                                    weights_dic['direct' + featuretyp + str(subgrams[-1])]['slice_inds'] = tmp_slice_inds  # same inds for all three
                                    weights_dic['direct' + featuretyp + str(subgrams[-1])]['vocabs'] = tmp_vocab
                        if not all_grams:
                            #we break here if we should do the grams we have been given at once
                            break
                    '''if we did the last of the above loop, then we have the data for e.g. char grams 1-5 or word grams 1-2.
                    That dataset we save so that we may be able to construct one large model
                    datainfo = {'data': dat, 'data_ids': new_ids, 'y_ids':y_ids, 'predictions': preds, 'y_org': y, 'predictions_cat':preds_cat}'''
                    # here we make our new model as soon as we have more than one featuretype; one dynAA model and one as a model taking all the input
                    if type(t_dat) == type(None):
                        #make inputs for dynAA and direct
                        t_dat = scipy.sparse.vstack([datainfo_train['data'], datainfo_val['data']], format='csr')
                        t_dat_ids = [datainfo_train['data_ids'].tolist() + datainfo_val['data_ids'].tolist()]
                        v_y_ids = [datainfo_val['y_ids']]
                        if len(datainfo_val['predictions'].shape) == 1:
                            v_pred = datainfo_val['predictions'].reshape((-1, 1))
                            tst_pred = datainfo_test['predictions'].reshape((-1, 1))
                        else:
                            v_pred = datainfo_val['predictions']
                            tst_pred = datainfo_test['predictions']
                        # make test set for both models
                        tst_dat = datainfo_test['data']
                        tst_dat_ids = [datainfo_test['data_ids']]
                        tst_y_ids = [datainfo_test['y_ids']]
                        t_weights = [datainfo_train['weights']]
                        t_vocab = [datainfo_train['vocabs']]
                        t_inds = [datainfo_train['slice_inds']]
                        tst_shape = [datainfo_test['data'].shape[1]]
                    else:
                        #add to inputs
                        t_dat_ids.append(datainfo_train['data_ids'].tolist() + datainfo_val['data_ids'].tolist())  #list of lists;
                        v_y_ids.append(datainfo_val['y_ids'])
                        #tmp has the result of the vstack and hstack of last iteration (that is the value we have to carry into our next iterateion of featuretypes)
                        t_dat = tmp
                        v_pred = tmp_y_pred
                        #add to test test
                        tst_dat = tst_tmp
                        tst_pred = tmp_y_pred_test
                        tst_dat_ids.append(datainfo_test['data_ids']) #same as assigning the tmp
                        tst_y_ids.append(datainfo_test['y_ids'])
                        t_weights.append(datainfo_train['weights'])
                        t_vocab.append(datainfo_train['vocabs'])
                        t_inds.append(datainfo_train['slice_inds'])
                        tst_shape.append(datainfo_test['data'].shape[1])

                #set back to none
                t_dat = None
                t_dat_ids = None
                v_y_ids = None
                v_pred = None
                # make test set for both models
                tst_dat = None
                tst_pred = None
                tst_dat_ids = None
                tst_y_ids = None
                t_weights = None
                t_vocab = None
                t_inds = None
                tst_shape = None
                gc.collect()

def train_comp(featuretyp:str, target:str, minTw_gram_auth, path:str, y:pd.DataFrame, id_subset:list=None, hash=False):
    # training of mdoels and saving of result to disk
    path = os.path.join(*ml_utils.split_path_unix_win(path))
    model_dir = os.path.join(path, featuretyp, str(minTw_gram_auth[0]), target)
    model_dir = ml_utils.make_save_dirs(model_dir)
    pred_path = ml_utils.make_prediction_dir(path, featuretyp, minTw_gram_auth[0], target)
    os.makedirs(pred_path, exist_ok=True)
    encoder_name = 'encoder_{}_{}_{}_authors.p'.format(target,minTw_gram_auth[0], minTw_gram_auth[2])
    #select subset for ids
    if id_subset:
        y = y.loc[y.uID.isin(id_subset), :]
    y_ids = y.uID.to_numpy().flatten()
    #vectorize input
    components, dat, ids = process_input(featuretyp, target, minTw_gram_auth, path, 'train', id_subset, hash=hash, base=False)

    #get slice_inds if tfidf
    slice_inds = [0, dat.shape[1]]
    vocabs = {}
    if 'tfidf' in components['vect_type']:
        #slice_inds = ml_utils.get_slice_inds(components['vectorizing'][0])
        slice_inds = ml_utils.get_slice_inds_from_feaure_inds(components['vectorizing'][0])
        vocabs = components['vectorizing'][0]
    #get ids as list
    if type(ids) == type({}):
        ids = ids[[key for key in ids.keys()][0]]

    ##assure the order of the entries
    dat, new_ids = ml_utils.assess_order(sorter=y_ids, data=dat, data_ids=np.array(ids).flatten(),
                                         return_ids=True)

    y_encoded, cats, target_key = encode_y(y, target, model_dir, encoder_name, id_subset, subset='train')

    gramstring = '_'.join([str(el) for el in minTw_gram_auth[1]])
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=123245, verbose=0)
  
    svm = ml_utils.train_scikit_model(model=svm, X_train=dat, y_train_enc=y_encoded, cats=cats, batch_size=1000, feature=featuretyp, grams=gramstring)
    print('make predictions...')
    sys.stdout.flush()
    preds, preds_cat = ml_utils.make_predictions(model=svm, X_test=dat, cats=cats, batch_size=1000)

    print('Train SVM Accurracy: {} ---- F1:{}'.format(accuracy_score(y[target_key].to_list(), preds_cat),
                                                 f1_score(y[target_key].to_list(), preds_cat, average='macro')))



    y[target + '_predicted_cat'] = preds_cat

    if not id_subset:
        print('save model to file...')
        sys.stdout.flush()
        with open(os.path.join(pred_path, "pred_{}_{}_{}_{}_{}_{}.json".format('train', target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2])), 'w') as w:
            y.to_json(w)


        with open(os.path.join(model_dir, 'svm_{}_{}_grams_{}_{}_{}_authors.p'.format(target, featuretyp,
                                                                                                   gramstring,
                                                                                                   minTw_gram_auth[0],
                                                                                                   minTw_gram_auth[2])), 'wb') as p:
            pickle.dump(svm, p)


    components['svm'] = svm
    datainfo = {'data_ids': new_ids, 'y_ids':y_ids, 'predictions': preds, 'y_org': y, 'vocabs':vocabs,
                'slice_inds': slice_inds,'predictions_cat':preds_cat, 'weights': svm.coef_, 'cats':cats, 'target_k':target_key}
    os.makedirs(os.path.join(pred_path, 'data'), exist_ok=True)
    dat_save_path = os.path.join(pred_path, 'data')
    if type(dat) == np.ndarray:
        dat_save_path = os.path.join(dat_save_path, "data_{}_{}_{}_{}_{}_{}.npy".format('train', target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        np.save(dat_save_path, arr=dat)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    elif type(dat) == scipy.sparse.csr.csr_matrix:
        dat_save_path = os.path.join(dat_save_path, "data_{}_{}_{}_{}_{}_{}.npz".format('train', target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        ml_utils.save_sparse(matrix=dat, path=dat_save_path)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    else:
        dat_save_path =os.path.join(dat_save_path,"data_{}_{}_{}_{}_{}_{}.p".format('train', target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        with open(dat_save_path, 'wb') as w:
            pickle.dump(dat, w)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    datainfo['data'] = dat_save_path
    #save to file
    with open(os.path.join(pred_path,"datainfo_{}_{}_{}_{}_{}_{}.p".format('train', target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2])), 'wb') as w:
        pickle.dump(datainfo, w)

    datainfo['data'] = dat
    return components, datainfo

def evaluate_comp(featuretyp:str, target:str, minTw_gram_auth, subset:str, path:str, y, id_subset:list=None,
                  components=None, load_components = False, hash=False, q=None):

    # evaluation on va- and train and saving of results to disk
    path = ml_utils.split_path_unix_win(path)
    path = os.path.join(*path)
    model_dir = ml_utils.make_save_dirs(os.path.join(path, featuretyp, str(minTw_gram_auth[0]), target))
    pred_path = ml_utils.make_prediction_dir(path, featuretyp, minTw_gram_auth[0], target)
    os.makedirs(pred_path, exist_ok=True)
    encoder_name = 'encoder_{}_{}_{}_authors.p'.format(target, minTw_gram_auth[0], minTw_gram_auth[2])

    #load components
    if not components and load_components:
        components = {}
        if featuretyp in ['num', 'polarity']:
            filen = 'pca_{}_{}_{}_authors'.format(featuretyp, str(minTw_gram_auth[0]), minTw_gram_auth[2])
            with open(os.path.join(model_dir, featuretyp, target,
                      'scaler_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])), 'wb') as p:
                vectorizer = pickle.load( p)
            components['vectorizers'] = [vectorizer]
        elif featuretyp == 'emoticon_c':
            gramsstring = '_'.join([str(el) for el in minTw_gram_auth[1]])
            filen = 'pca_{}_grams_{}_{}_{}_authors.p'.format(featuretyp, gramsstring, str(minTw_gram_auth[0]),
                                                             minTw_gram_auth[2])
            with open(os.path.join(model_dir, featuretyp, target,
                      'vectorizer_{}_{}_{}_authors.p'.format(featuretyp, minTw_gram_auth[1], minTw_gram_auth[2])),
                      'rb') as p:
                vectorizer = pickle.load(p)
            components['vectorizers'] = [vectorizer]
        else:
            gramsstring = '_'.join([str(el) for el in minTw_gram_auth[1]])

            filen = 'pca_{}_grams_{}_{}_{}_authors.p'.format(featuretyp, gramsstring, minTw_gram_auth[0],
                                                             minTw_gram_auth[2])
            vectorizer_dic = ml_utils.load_make_vectorizers(target, path, featuretyp, minTw_gram_auth, hash)
            with open(os.path.join(model_dir, featuretyp, target,
                                   'tfidf_{}_grams_{}_{}_{}_hash_{}.p'.format(featuretyp, gramsstring, minTw_gram_auth[0],
                                                                              minTw_gram_auth[2] ,hash)), 'rb') as p:
                trans = pickle.load(p)
            components['vectorizers'] = [vectorizer_dic, trans]
    gramstring = '_'.join([str(el) for el in minTw_gram_auth[1]])



    components, dat, ids = process_input(featuretyp=featuretyp, target=target, minTw_gram_auth=minTw_gram_auth,
                  path=path, subset=subset, id_subset=id_subset, components=components)

    #get slice_inds if tfidf
    slice_inds = [0, dat.shape[1]]
    if 'tfidf' in components['vect_type']:
        slice_inds = ml_utils.get_slice_inds(components['vectorizing'][0])

    if type(ids) == type({}):
        ids = ids[[key for key in ids.keys()][0]]
    if id_subset:
        y = y.loc[y.uID.isin(id_subset), :]
    y_ids = y.uID.to_numpy().flatten()

        ##assure the order of the entries

    dat, new_ids = ml_utils.assess_order(sorter=y_ids, data=dat, data_ids=np.array(ids).flatten(),
                                                 return_ids=True)
    y_encoded, cats, target_key = encode_y(y, target, model_dir, encoder_name, id_subset, subset=subset)
    #eval = components['log_model'].evaluate(dat, y_encoded, verbose=2,)
    #print('Results of model evaluation: {}'.format(eval[0]))
    print('making predictions...')

    sys.stdout.flush()

    with open(os.path.join(model_dir, 'svm_{}_{}_grams_{}_{}_{}_authors.p'.format(target, featuretyp,
                                                                                     gramstring,
                                                                                     minTw_gram_auth[0],
                                                                                     minTw_gram_auth[2])), 'rb') as p:
        svm = pickle.load(p)

    preds, preds_cat = ml_utils.make_predictions(model=svm, X_test=dat, cats=cats, batch_size=1000)

    print('{} SVM Accurracy: {} ---- F1:{}'.format(subset, accuracy_score(y[target_key].to_list(), preds_cat),
                                                 f1_score(y[target_key].to_list(), preds_cat, average='macro')))


    y[target + '_predicted_cat'] = preds_cat

    if not id_subset:
        print('saving predictions...')

        with open(os.path.join(pred_path, "pred_{}_{}_{}_{}_{}_{}.json".format(subset, target, featuretyp, gramstring, minTw_gram_auth[0], minTw_gram_auth[2])), 'w') as w:
            y.to_json(w)

    components['svm'] = svm# logistic_model
    datainfo = {'data_ids': new_ids, 'y_ids':y_ids, 'predictions': preds, 'y_org': y,
                'slice_inds': slice_inds,'predictions_cat':preds_cat, 'weights': svm.coef_, 'cats':cats, 'target_k':target_key}
    os.makedirs(os.path.join(pred_path, 'data'), exist_ok=True)
    dat_save_path = os.path.join(pred_path, 'data')
    if type(dat) == np.ndarray:
        dat_save_path = os.path.join(dat_save_path, "data_{}_{}_{}_{}_{}_{}.npy".format(subset, target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        np.save(dat_save_path, arr=dat)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    elif type(dat) == scipy.sparse.csr.csr_matrix:
        dat_save_path = os.path.join(dat_save_path, "data_{}_{}_{}_{}_{}_{}.npz".format(subset, target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        ml_utils.save_sparse(matrix=dat, path=dat_save_path)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    else:
        dat_save_path =os.path.join(dat_save_path,"data_{}_{}_{}_{}_{}_{}.p".format(subset, target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2]))
        with open(dat_save_path, 'wb') as w:
            pickle.dump(dat, w)
        print('saved dat separately to {}...'.format(dat_save_path))
        sys.stdout.flush()
    datainfo['data'] = dat_save_path
    #save to file
    with open(os.path.join(pred_path,"datainfo_{}_{}_{}_{}_{}_{}.p".format(subset, target, featuretyp, gramstring, minTw_gram_auth[0],
                                                                    minTw_gram_auth[2])), 'wb') as w:
        pickle.dump(datainfo, w)

    datainfo['data'] = dat

    if type(q) != type(None):

        row = evaluations.make_score_table(info=[target, minTw_gram_auth[0], minTw_gram_auth[2], featuretyp, minTw_gram_auth[1]],
                                     y_true=y[target_key].to_numpy(), y_pred=np.array(preds_cat).flatten(), path=path,
                                     modeltype='basline', mode='return')

        q.put(row)
        print('put into queue...')
        sys.stdout.flush()
    return datainfo

def process_wrapper(featuretyp:str, target:str, minTw_gram_auth, path:str, q=None, id_subset=[], hash=False):
    # wrapper function for training process so that multiprocessing becomes possible
    import tensorflow as tf
    pp = ml_utils.split_path_unix_win(path)
    id_type = ['gender_ids', 'age_ids']
    idt = id_type[0]
    for el in id_type:
        if target in el:
            idt=el
    print('Doing {}'.format(idt))
    sys.stdout.flush()
    pid = mp.current_process().pid


    print('{} is doing target {} on features {} for length {} with grams {} and {} authors'.format(pid, target,
                                                                                                   featuretyp,
                                                                                                   minTw_gram_auth[0], '_'.join([str(el) for el in minTw_gram_auth[1]]),
                                                                                                   minTw_gram_auth[2]))
    sys.stdout.flush()

    subset='train'
    y_path = os.path.join(*pp, idt, subset,  '{}_ids_{}_{}_{}_authors_balanced.json'.format(subset, target,
                                                                                            minTw_gram_auth[0],minTw_gram_auth[2]))

    y = pd.read_json(y_path)

    components = train_comp(featuretyp=featuretyp, target=target, minTw_gram_auth=minTw_gram_auth,
                             path=path, y=y, id_subset=id_subset)
    #
    subset='val'
    y = pd.read_json(os.path.join(*pp, idt, subset,  '{}_ids_{}_{}_{}_authors_balanced.json'.format(subset, target,
                                                                                                             minTw_gram_auth[0],minTw_gram_auth[2])))
    components = evaluate_comp(featuretyp=featuretyp, target=target, minTw_gram_auth=minTw_gram_auth,
                                path=path, y=y, id_subset=id_subset, subset=subset, q=None)
    #
    subset ='test'
    y = pd.read_json(os.path.join(*pp, idt, subset,  '{}_ids_{}_{}_{}_authors_balanced.json'.format(subset, target,
                                                                                                             minTw_gram_auth[0],minTw_gram_auth[2])))
    components = evaluate_comp(featuretyp=featuretyp, target=target, minTw_gram_auth=minTw_gram_auth,
                                path=path, y=y, id_subset=id_subset, subset=subset, q=q)

    print('{} is finished with target {} on features {} for length {} with grams {} and {} authors'.format(pid, target,
                                                                                                   featuretyp,
                                                                                                   minTw_gram_auth[
                                                                                                       0], '_'.join([str(el) for el in minTw_gram_auth[1]]),
                                                                                                   minTw_gram_auth[
                                                                                                       2]))
    sys.stdout.flush()

def process_wrapper_dynAA(path:str, subdic:dict, dynAA:bool, target:str, true_train:bool, q=None, singular=False):
    # wrapper function for dynAA training process so that multiprocessing becomes possible

    pp = os.path.join(*ml_utils.split_path_unix_win(path))

    pid = mp.current_process().pid
    types = list(subdic.keys())
    print('{} is doing target {} on features {} for length {} with grams {} and {} authors'.format(pid, target,
                                                                                                   types,
                                                                                                   subdic[types[-1]][0],
                                                                                                   '_'.join(
                                                                                                       [str(el) for el
                                                                                                        in subdic[types[-1]][1]]),
                                                                                                   subdic[types[-1]][2]))
    sys.stdout.flush()
    partial = False
    train_loop(path=pp, in_dic=subdic, targets=target, dynAA=dynAA, true_train=False, all_grams=True, q=q, partial=partial, singular=singular)
    gc.collect()
    print('{} is done with previous job...'.format(pid))
    sys.stdout.flush()



if __name__ == '__main__':

    path = input("Enter the path to the file: ")
    test = input('Enter whether we are in the testing phase: [True, False]')
    if test in ['True', 'False']: #sanity check
        test = ast.literal_eval(test)
    else:
        print('Invalid input for test')
        sys.exit(1)
    random.seed(123)
    featuretyp_dic = {'emoticon_c': [[100, 250, 500], [1], [1000, 500, 150, 50]],
                        'char': [[100, 250, 500], list(range(2, 5 + 1)), [1000, 500, 150, 50]],
                      'word': [[100, 250, 500], list(range(1, 2 + 1)), [1000, 500, 150, 50]],
                      'tag': [[100, 250, 500], list(range(1, 3 + 1)), [1000, 500, 150, 50]],
                      'dep': [[100, 250, 500], list(range(1, 3 + 1)), [1000, 500, 150, 50]],
                      'asis': [[100, 250, 500], list(range(2, 5 + 1)), [1000, 500, 150, 50]],
                      'lemma': [[100, 250, 500], list(range(1, 2 + 1)), [1000, 500, 150, 50]],
                      'dist': [[100, 250, 500], list(range(2, 5 + 1)), [1000, 500, 150, 50]],
                      'pos': [[100, 250, 500], list(range(1, 3 + 1)), [1000, 500, 150, 50]],
                      'num': [[100, 250, 500], [1], [1000, 500, 150, 50]],
                      'polarity': [[100, 250, 500], [1], [1000, 500, 150, 50]]
                      }
    targets_all = ['gender', 'age']
    order = ['dist', 'char', 'asis', 'pos', 'tag', 'dep', 'lemma', 'word','emoticon_c','polarity', 'num']
    #path = '../../Data/pan19-celebrity-profiling-training-dataset-2019-01-31/preprocessed/workset/creator'
    #sys.argv.extend(['10', 'dynAA', 'age', '100', '250', '500'])
    if not test:
        ncpus = int(sys.argv[1])

        pool = mp.Pool(ncpus)
        manager = mp.Manager()
        q = manager.Queue()
        jobs = []
        result_dirs = {}
        # different options what this script is able to do
        if str(sys.argv[2]) == 'precalc':
            #here we precalculate the feature enigneering so that theys may be laoded during training (saves lots of time later)
            task = process_wrapper
            if len(sys.argv) >3:
                featuretyp_dic = {key:featuretyp_dic[key] for key in sys.argv[3:]}
            for target in targets_all:
                result_dirs[target] = ml_utils.make_result_dirs(os.path.join(path), target)
                for featuretype in featuretyp_dic.keys():
                    for minTW in featuretyp_dic[featuretype][0]:
                        for numAuth in sorted(featuretyp_dic[featuretype][2]):
                            minTw_gram_auth = [minTW, featuretyp_dic[featuretype][1], numAuth]
                            for i in range(1, len(minTw_gram_auth[1]) + 1):
                                subgrams = copy.deepcopy(minTw_gram_auth[1][0:i])
                                sub_minTw_gram_auth = copy.deepcopy(minTw_gram_auth)
                                sub_minTw_gram_auth[1] = subgrams

                                jobs.append((featuretype, target, sub_minTw_gram_auth ,path, q))

            random.shuffle(jobs)

        elif str(sys.argv[2]) == 'dynAA_large':
            # makes a full dynAA calcualtion on all possible combinations using all feature types at once
            targets_all = np.array([sys.argv[3].split('_')]).flatten().tolist()
            for target in targets_all:
                result_dirs[str(target)] = ml_utils.make_result_dirs(os.path.join(path), str(target))
            minTweets = [int(el) for el in sys.argv[4:]]
            task = process_wrapper_dynAA
            for minTw in minTweets:
                #iterate over minTweet lenghts
                for i in range(1, len(order)+1):
                    #iterate over featuretype combinations
                    subTypes = copy.deepcopy(order[0:i])
                    numAuths = sorted(featuretyp_dic[order[0]][2])
                    sub_dic = {}
                    curr_t = []
                    for typ in subTypes:
                        #curr_t.append(type)
                        #make our dic for passing
                        sub_dic[typ] = copy.deepcopy(featuretyp_dic[typ])
                        sub_dic[typ][0] = [minTw]
                    grams = copy.deepcopy(sub_dic[subTypes[-1]][1])
                    for k in range(1, len(grams) + 1):
                        '''iterate over slices of subtypes for last type. ie if
                        subtypes:= [char, word] then we iterate over ngrams for word [1,2] because
                        type block before (i.e. char) gets always full ngram type set'''
                        sub_dic[subTypes[-1]][1] = grams[0:k]
                        for j in range(1, len(numAuths)):
                            #iterate over pairs over numAuthors
                            auThSlice = copy.deepcopy(numAuths[(j-1):(j+1)])
                            for el in sub_dic.keys():
                                sub_dic[el][2] = auThSlice
                            for dynAA in [True, False]:
                                job = (path, sub_dic, dynAA, targets_all, True, q)
                                jobs.append(job)

        elif str(sys.argv[2]) == 'dynAA':
            '''makes a partial dynAA calculation using only a subset of the tasks and features
            (basically task partitioning on a manual level and if not all results are needed)'''
            targets_all = np.array([sys.argv[3].split('_')]).flatten().tolist()
            minTweets = [int(el) for el in sys.argv[4:]]
            task = process_wrapper_dynAA
            for targets in targets_all:
                result_dirs[str(targets)] = ml_utils.make_result_dirs(os.path.join(path), str(targets))
                targets = np.array([targets]).copy().tolist()
                for minTw in minTweets:
                    #iterate over minTweet lenghts
                    sub_dic = copy.deepcopy(featuretyp_dic)
                    numAuths = sorted(featuretyp_dic[order[0]][2])
                    for j in range(1, len(numAuths)):
                        #iterate over pairs over numAuthors
                        auThSlice = copy.deepcopy(numAuths[(j-1):(j+1)])
                        for typ in sub_dic.keys():
                            sub_dic[typ][2] = auThSlice
                            sub_dic[typ][0] = [minTw]
                        for dynAA in ['both']:
                            passer = copy.deepcopy(sub_dic)
                            job = (path, passer, dynAA, targets, True, q)
                            jobs.append(job)


        elif str(sys.argv[2]) == 'dynAA_red':
            # does dynAA only on the feature types distortion, word, character, asis and lemma (most informative ones)
            targets_all = np.array([sys.argv[3].split('_')]).flatten().tolist()
            minTweets = [int(el) for el in sys.argv[4:]]
            task = process_wrapper_dynAA
            for targets in targets_all:
                result_dirs[str(targets)] = ml_utils.make_result_dirs(os.path.join(path), str(targets))
                targets = np.array([targets]).copy().tolist()
                for minTw in minTweets:
                    #iterate over minTweet lenghts
                    sub_dic = copy.deepcopy(featuretyp_dic)
                    sub_dic = {k:v for k,v in sub_dic.items() if k in ['dist','word', 'char', 'asis', 'lemma']}
                    #sub_dic = {k: v for k, v in sub_dic.items() if k in [ 'word', 'char', 'asis', 'lemma']}
                    numAuths = sorted(featuretyp_dic[order[0]][2])
                    for j in range(1, len(numAuths)):
                        #iterate over pairs over numAuthors
                        auThSlice = copy.deepcopy(numAuths[(j-1):(j+1)])
                    #for auThSlice in [[50, 150]]:
                        for typ in sub_dic.keys():
                            sub_dic[typ][2] = auThSlice
                            sub_dic[typ][0] = [minTw]
                        for dynAA in ['both']:
                            passer = copy.deepcopy(sub_dic)
                            job = (path, passer, dynAA, targets, True, q)
                            jobs.append(job)
                            print(sub_dic)
                            sys.stdout.flush()

            featuretyp_dic = sub_dic

        elif str(sys.argv[2]) == 'dynAA_singular':
            # dynAA only for a singular feature type
            targets_all = np.array([sys.argv[3].split('_')]).flatten().tolist()
            minTweets = [int(el) for el in sys.argv[4:]]
            task = process_wrapper_dynAA
            for targets in targets_all:
                result_dirs[str(targets)] = ml_utils.make_result_dirs(os.path.join(path), str(targets))
                targets = np.array([targets]).copy().tolist()
                for minTw in minTweets:
                    for el in order:
                        sub_dic = {}
                        sub_dic[el] = copy.deepcopy(featuretyp_dic[el])
                        numAuths = sorted(featuretyp_dic[order[0]][2])
                        for j in range(1, len(numAuths)):
                            #iterate over pairs over numAuthors
                            auThSlice = copy.deepcopy(numAuths[(j-1):(j+1)])
                            sub_dic[el][2] = auThSlice
                            sub_dic[el][0] = [minTw]
                            passer = copy.deepcopy(sub_dic)
                            job = (path, passer, 'both', targets, True, q, True)
                            jobs.append(job)
                            print(sub_dic)
            featuretyp_dic = {'s':1, 'i':2, 'n':3, 'g':4,'u':5, 'l':6, 'a':7, 'r':8}

        # here we start our multiprocessing after having constructed the task array above
        procs = []

        for job in jobs:

            proc = pool.apply_async(task, job)
            procs.append(proc)
        # collect results from the workers through the pool result queue
        print('collect results from job cycle...')
        sys.stdout.flush()
        todos = len(jobs)
        for i in range(0, todos):
            tmp = procs.pop(0)
            tmp.get()
            del tmp
        print('sleep after cycle...')
        sys.stdout.flush()
        print('closing down the pool and exit :)')
        sys.stdout.flush()
        pool.close()
        pool.join()

        str_add = ''.join([str(key)[0] for key in featuretyp_dic.keys()])
        str_add = str_add + '_' + '_'.join([str(minTW) for minTW in minTweets]) + '_'

        for target in targets_all:
            result_dirs[target + '_distortion'] = jsonlines.open(os.path.join(result_dirs[target], str_add + 'distortion.ndjson'), mode='w')
            result_dirs[target] = jsonlines.open(os.path.join(result_dirs[target], str_add+'basic_scores.ndjson'), mode='w')


        count = 0
        while q.empty() != True:
            res = q.get()
            count +=1
            if type(res) != type(None):
                ext = ''
                print((count)/2)
                if sum(["spearman" in el.lower() for el in res.keys()]):
                    ext = '_distortion'
                result_dirs[res['target']+ext].write(res)

        for target in targets_all:
            result_dirs[target].close()
            result_dirs[target + '_distortion'].close()




    else:
        #testing pruposes
        minTw_gram_auth = [500, [1,2], 100]
        pp = ml_utils.split_path_unix_win(path)
        id_type = ['gender_ids', 'age_ids']
        featuretyp_dic = { 'char': [[500], list(range(2, 5 + 1)), sorted([1000, 500, 150, 50])],
                            'word': [[500],[1,2], sorted([1000, 500, 150, 50])],
                           'num': [[500],[1], sorted([1000, 500, 150, 50])]}
        train_loop(path=path, targets='gender', in_dic =featuretyp_dic, dynAA = True, hash = False, true_train =False, all_grams=False, partial=True, baseline_fetch=True)
