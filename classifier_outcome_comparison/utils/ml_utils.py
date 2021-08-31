import os
import pkg_resources
pkg_resources.require("Scipy==1.7.0")
import scipy
import sys
sys.path.append('./')
import generator
import tqdm
import copy
import gc
import re
import json
import jsonlines
import pickle
import pandas as pd
#import tables
from scipy import sparse
from scipy.special import softmax
import numpy as np
import math
import random as rd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

'''def store_sparse_mat(M, name, filename='store.h5'):
    """
    Store a csr matrix in HDF5

    Parameters
    ----------
    M : scipy.sparse.csr.csr_matrix
        sparse matrix to be stored

    name: str
        node prefix in HDF5 hierarchy

    filename: str
        HDF5 filename
    """
    assert(M.__class__ == sparse.csr.csr_matrix), 'M must be a csr matrix'
    with tables.open_file(filename, 'a') as f:
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            full_name = f'{name}_{attribute}'

            # remove existing nodes
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            # add nodes
            arr = np.array(getattr(M, attribute))
            atom = tables.Atom.from_dtype(arr.dtype)
            ds = f.create_carray(f.root, full_name, atom, arr.shape)
            ds[:] = arr'''

'''def load_sparse_mat(name, filename='store.h5'):
    """
    Load a csr matrix from HDF5

    Parameters
    ----------
    name: str
        node prefix in HDF5 hierarchy

    filename: str
        HDF5 filename

    Returns
    ----------
    M : scipy.sparse.csr.csr_matrix
        loaded sparse matrix
    """
    with tables.open_file(filename) as f:

        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, f'{name}_{attribute}').read())

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M'''

def chunkify(linebytes, chunks):

    chunksize = math.floor(len(linebytes)/chunks)
    if chunksize == 0:
        chunksize = 1
    linebytes = [linebytes[i:(i+chunksize)] for i in range(0, len(linebytes), chunksize)]

    return linebytes

def split_path_unix_win(path):
    #make windows-unix problem go away
    if  '\\' in path:
        path = path.split('\\')
    elif '/' in path:
        path = path.split('/')

    return path

def batch_read(fileobj, batchsize, to_json=True):
    lines = []
    for _ in range(batchsize):
        line = fileobj.readline()
        if line and to_json:
            line = jsonize(line)
            line = make_uids(line, 'tweetIDs', secondary='tweetID')
            lines.append(line)
        elif line:
            lines.append(line)

    return lines

def jsonize(line):
    try:
        line = json.loads(line)
    except:
        try:
            line = line.split('{')[1]
            line = line.split('}')[0]
            line = json.loads('{'+line+'}')
        except:
            print('manual line loading failed as well...')
            sys.stdout.flush()
            return False

    return line

def make_uids(dic, key, secondary='tweetID'):
    if key not in dic.keys():
        key = secondary
    dic['uID'] = '|'.join(dic[key])
    return dic

def select_uids(df, ids):
    df = df.loc[df['uID'].isin(ids),:]
    return df

def pandas_to_ndjson(df:pd.DataFrame, f):
    df = df.to_dict(orient='records')
    writer = jsonlines.Writer(f)
    writer.write_all(df)
    sys.stdout.flush()
    writer.close()

def make_save_dirs(path):
    # create files for later saving
    path = split_path_unix_win(path)
    ind = path.index('preprocessed')
    model_dir = os.path.join(*(path[0:ind]+ ['models'] + path[(ind+1):]))
    os.makedirs(model_dir, exist_ok=True)
    print('made models directory with subdirs...')
    sys.stdout.flush()
    return model_dir

def make_result_dirs(path, target):
    # create files for later saving
    path = split_path_unix_win(path)
    ind = path.index('preprocessed')
    model_dir = os.path.join(*(path[0:ind] +['results'] + path[(ind+1):]), target)
    os.makedirs(model_dir, exist_ok=True)
    print('made result directory with subdirs...')
    sys.stdout.flush()
    return model_dir

def identity_prepr(text):
    return text

def drop_duplicates(ids:list, csr:sparse.csr_matrix):
    id_positions = {}
    #make dictionary with pairs uID:index
    #always keep the first of them
    for i in range(0, len(ids)):
        id_positions[ids[i]] = id_positions.get(ids[i], i)

    #make a list from the values
    inds = [v for v in id_positions.values()]
    #sort in ascending so that they are all in the initial order again
    inds.sort()
    #make the new ids in correct order
    ids = [ids[i] for i in range(0, len(inds))]
    csr = csr[inds, :]
    return ids, csr


def duplicate_uid_tester(current:pd.DataFrame, idset:set, key='uID'):
    #make sure we do not have duplicates in this batch of data
    #we also reset the index here so that everything is in order from 0 to n (that is the reason why iloc works)
    current = current.drop_duplicates(subset=[key], keep='first').reset_index(drop=True)
    #check which ones are not duplicates to batches which came before
    loc = current.columns.get_loc(key)+1
    inds = []
    for row in current.itertuples():
        #test whether we have seen id before - if not add to our checker set and index in df to inds
        if row[loc] not in idset:
            idset.update(row[loc])
            inds.append(row[0])
    #subset df so that we have only the indices in df which are in inds
    current = current.iloc[inds]
    current.reset_index(drop=True, inplace=True)

    return current, idset

def duplicate_fast_uid_tester(uids):
    reduced= {}

    for i, id in enumerate(uids):
        reduced[id] = reduced.get(id, i)

    returner = [[],[]]
    for k, v in reduced.items():
        returner[0].append(k)
        returner[1].append(v)

    return returner[0], returner[1]

def reset_num(num:pd.DataFrame, fix:pd.DataFrame):
    num = num.sort_values(by=['uID']) #bring them into the same order so that the correct values are written into each row lateron
    fix = fix.copy(deep=True).sort_values(by=['uID'])
    fix_sub = fix.loc[fix.uID.isin(num.uID), :]
    #fix emoji count
    num.loc[num.uID.isin(fix.uID), ['emojis_num']] = fix_sub.apply(lambda x: len(x['emoji'])/len(x['uID'].split('|')), axis=1).to_list()
    num.loc[num.uID.isin(fix.uID), ['emotic_num']] = fix_sub.apply(lambda x: len(x['emoticon'])/len(x['uID'].split('|')), axis=1).to_list()
    num.sort_index(inplace=True)

    return num

def identity_tokenizer(text):
    return text

def identity_analyzer(text):
    return text

def zero_columns(M, columns):
    diag = sparse.eye(M.shape[1]).tolil()
    for c in columns:
        diag[c, c] = 0
    return M.dot(diag)

def make_vocab(dict, path):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(dict, json_file, ensure_ascii=False)
    print('saved vocab to disk...')
    sys.stdout.flush()

def get_vocab(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        vocab = json.load(json_file)

    return vocab

def update_vocab(vocab, lines:pd.DataFrame, key):
    numerics = re.compile('[0-9]')
    for el in lines[key]:
        subvocab = {}
        #basically make a set here - we do not want total frequency but doc frequency (i.e. in how many docs does it appear not how often does it appear overall
        for token in el:
            #make sure that we do not have numerics in there...
            if not re.search(numerics, token):
                subvocab[token] = subvocab.get(token, 0) + 1

        for token in subvocab.keys():
            vocab[token] = vocab.get(token, 0) +1
    return vocab

def update_vocab2(vocab, lines:pd.DataFrame, key):
    for el in lines[key]:
        subvocab = {}
        #basically make a set here - we do not want total frequency but doc frequency (i.e. in how many docs does it appear not how often does it appear overall
        for token in el:
            subvocab[token] = subvocab.get(token, 0) + 1

        for token in subvocab.keys():
            vocab[token] = vocab.get(token, (0,0))
            vocab[token][0] += 1
            vocab[token][1] +=subvocab[token]
    return vocab

def reduce_vocab2(vocab, cutoff):
    print('Omitting all features appearing less than {} times'.format(cutoff))
    sys.stdout.flush()
    ind = 0
    new = {}
    old_vocab = 0
    for key, value in vocab.items():
        old_vocab +=1
        if value[0] >=cutoff:
            new[key] = ind
            ind +=1
    print('Vocab had {} entries and now has {} entries'.format(old_vocab, ind))
    sys.stdout.flush()
    return new, ind


def reduce_vocab_max_features(vocab, max_features:int):
    print('Only keeping the {} most common features...'.format(max_features))
    sys.stdout.flush()
    features = []
    appreances = []
    for key, value in vocab.items():
        features.append(key)
        appreances.append(value[1])
    inds = np.argsort(appreances)[-max_features:]
    features = np.array(features)[inds]


def reduce_vocab(vocab, cutoff):
    print('Omitting all features appearing less than {} times'.format(cutoff))
    sys.stdout.flush()
    ind = 0
    new = {}
    old_vocab = 0
    for key, value in vocab.items():
        old_vocab +=1
        if value >=cutoff:
            new[key] = ind
            ind +=1
    print('Vocab had {} entries and now has {} entries'.format(old_vocab, ind))
    sys.stdout.flush()
    return new, ind

def vocab_maker(vocab):
    ind = 0
    new = {}
    for key in vocab.keys():
        new[key] = ind
        ind +=1

    return new, ind

def construct_tdm(docs, vocab, maxcol):
    indptr = [0]
    indices = []
    data = []
    exception_counter = 0
    term_counter = 0
    for d in docs:
        for term in d:
            term_counter +=1
            index = vocab.get(term, 'exception') #if term does not appear often enough
            if index == 'exception':
                exception_counter +=1
                continue
            indices.append(index)
            data.append(1)
        #make it so that we have enough columns
        indices.append(maxcol+1)
        data.append(1)
        indptr.append(len(indices))
        print('went over {} terms'.format(term_counter))
        print('of these {} are not in dict'.format(exception_counter))
        sys.stdout.flush()

    return sparse.csr_matrix((data, indices, indptr), dtype=int)

def find_variance_cutoff(pca, dat, cutoff=0.99, tol = 0.005, start = 100):
    #recurisve cutoff search (divide and conquer)
    print('make pca with {} components'.format(start))
    sys.stdout.flush()
    pca = pca.set_params(n_components=start)
    #pca = pca.set_params(batch_size = round(start*3))
    pca = pca.fit(dat)
    diff = np.abs(pca.explained_variance_ratio_.sum() - cutoff)
    print('Difference to cutoff is {}'.format(diff))
    sys.stdout.flush()
    if diff < tol:
        print('Use {} components. That is {} of num_features'.format(start, round(start/dat.shape[1], 2)))
        sys.stdout.flush()
        return pca

    #else we continue search recursively
    if pca.explained_variance_ratio_.sum() > cutoff:
        start = round(3*start/4)
        return find_variance_cutoff(pca, dat, cutoff, tol, start = start)

    else:
        start = start + round((dat.shape[1]-start)/2)
        return find_variance_cutoff(pca, dat, cutoff, tol, start=start)

def make_gram_dict(path:str):

    gram_dict = {}
    path = os.path.join(*split_path_unix_win(path))

    for dir in os.listdir(path):
        if dir not in ['vectors', 'process', 'age_ids', 'gender_ids']:
            direct = os.path.join(path, dir)
            if os.path.isdir(direct):
                #make key char, num, etc.
                gram_dict[dir] = {}
                for tweetLen in os.listdir(direct):
                    subdirect = os.path.join(direct, tweetLen)
                    if os.path.isdir(subdirect):
                        #make key tweetlen 500, 250, 100
                        gram_dict[dir][str(tweetLen)] = {}
                        for target in os.listdir(subdirect):
                            target_direct = os.path.join(subdirect, target)
                            if os.path.isdir(target_direct):
                                #make target key
                                gram_dict[dir][str(tweetLen)][target] = {}
                                for subset in os.listdir(target_direct):
                                    subset_dir = os.path.join(target_direct, subset)
                                    if os.path.isdir(subset_dir):
                                        #make subset key, i.e. train, test, val
                                        gram_dict[dir][str(tweetLen)][target][subset] = {}
                                        #now we only have files (sometimes wtih and sometimes without grams
                                        files = os.listdir(subset_dir)
                                        for file in files:
                                            ##now we have to go over the number of authors in each subset
                                            authors = file.split('_')[-2]
                                            #make or keep key with dictionary
                                            gram_dict[dir][str(tweetLen)][target][subset][authors] = gram_dict[dir][str(tweetLen)][target][subset].get(authors, {})
                                            if '_grams_' in file:
                                                grams = file.split('_')[-4]
                                                gram_dict[dir][str(tweetLen)][target][subset][authors][grams] = {}
                                                #gram_dict[dir][str(tweetLen)][target][subset][authors][grams]['file'] = os.path.join(subset_dir, file)
                                                gram_dict[dir][str(tweetLen)][target][subset][authors][grams]['numGrams'] = int(grams)

                                            else:
                                                gram_dict[dir][str(tweetLen)][target][subset][authors]['no_gram'] = {}
                                                #gram_dict[dir][str(tweetLen)][target][subset][authors]['no_gram']['file']= os.path.join(subset_dir, file)
                                                gram_dict[dir][str(tweetLen)][target][subset][authors]['no_gram']['numGrams'] = 1

                                            gram_dict[dir][str(tweetLen)][target][subset][authors]['ids'] = os.path.join(path, '{}_ids'.format(target),
                                                                                                                            subset, '{}_ids_{}_{}_{}_authors_balanced.json'.format(subset,
                                                                                                                                                                                    target,
                                                                                                                                                                                    tweetLen,
                                                                                                                                                                                   authors))

    return gram_dict

def assess_equal(dat1, ids1, dat2, ids2):
    sym = set(ids1).symmetric_difference(set(ids2))


    if len(ids1) > len(ids2):
        assert len(set(ids2) - set(ids1)) == 0

        print('data1 is ({}) larger than dat2 ({})...'.format(dat1.shape, dat2.shape))
        sys.stdout.flush()
        dat1 = assess_order(np.array(ids2), dat1, np.array(ids1))
        ids1 = ids2
    elif len(ids1) < len(ids2):
        assert len(set(ids1) - set(ids2)) == 0
        print('data2 is ({}) larger than dat1 ({})...'.format(dat2.shape, dat1.shape))
        sys.stdout.flush()
        dat2 = assess_order(np.array(ids1), dat2, np.array(ids2))
        ids2 = ids1

    else:
        ##assert that the ids follow the same order
        if sum([ids1[i] == ids2[i] for i in range(0, len(ids1))]) == len(ids1):
            print('ids are identical...')
            sys.stdout.flush()

        else:
            #call the resort id funtion and append a nonsense id to one of the two so that we resort
            print('num ids have the same length, but they are not identical in content - make intersection and resort...')
            sys.stdout.flush()
            intersect = list(set(ids1).intersection(set(ids2)))
            dat1 = assess_order(np.array(intersect), dat1, np.array(ids1))
            dat2 = assess_order(np.array(intersect), dat2, np.array(ids2))
            ids1 = intersect
            ids2 = intersect
    return dat1, ids1,dat2, ids2

def assess_order(sorter,data, data_ids, return_ids=False):

    if type(sorter) == type({}):
        sorter = sorter[list(sorter.keys())[0]]

    sorter = np.array(sorter).flatten()

    d = {}
    for ind, uID in enumerate(data_ids):
        d[uID] = d.get(uID, ind)
    sort_ids = [d[sorter[i]] for i in range(0, sorter.shape[0])] #order we want the ids in
    data = data[sort_ids, :]  #slice the array in correct order
    assert len(sort_ids) == sorter.shape[0]
    assert len(sort_ids) == data.shape[0]
    if not return_ids:
        return data
    else:
        return data, sorter

def asses_same_author(ids, column=None, path:str=None, return_labels=True):
    if return_labels:
        y = pd.read_json(path).set_index('id', inplace=False)
    ids = []
    for uID in ids:
        tweets = uID.split('|')
        tester_dic = {tweets[0].split('_')[0]: 'authID'}
        if len(tweets) >1:
            for i in range(1, len(tweets)):
                #test wether key exists, i.e. whether all authors in concated tweets are the same; if fail, then not (should never fail - grevious exception)
                tmp = tester_dic[tweets[1].split('_')[0]]
        ids.append(list(tester_dic.keys())[0])
    if return_labels:
        y = y.loc[ids].reset_index(inplace=False)
        return y[column]

def load_make_vectorizers(target, path, featuretyp, minTw_gram_auth, hash):
    vectorizer_dic = {}
    minTW = minTw_gram_auth[0]
    numAuthors = minTw_gram_auth[2]
    path_vocab = os.path.join(*split_path_unix_win(path),
                              featuretyp, str(minTW), 'vocab', target)
    for gram in sorted(minTw_gram_auth[1]):
        vectorizer_dic[str(gram)] = {}

        if not hash:
            path_vocab_red = os.path.join(path_vocab,
                                      '{}_grams_{}_{}_{}_authors_reduced.vocab'.format(featuretyp, gram, minTW,
                                                                                       numAuthors))
            vocab = get_vocab(path_vocab_red)
            vectorizer_dic[str(gram)]['vect'] = CountVectorizer(vocabulary=vocab, analyzer=identity_analyzer)
            vectorizer_dic[str(gram)]['vocab'] = vocab
        else:
            path_colids = os.join.path(path_vocab,
                                       '{}_grams_{}_{}_{}_authors_reduced.colids'.format(featuretyp, gram, minTW,
                                                                                         numAuthors))
            vectorizer_dic[str(gram)]['vect'] = HashingVectorizer(analyzer=identity_analyzer)
            with open(path_colids, 'rb') as p:
                    colinds = pickle.load(p)

            vectorizer_dic[str(gram)]['colinds'] = colinds

    return vectorizer_dic

def make_prediction_dir(path, featuretype, minTweet, target):
    path = split_path_unix_win(path)
    ind = path.index('preprocessed')
    path[ind] = 'predictions'
    path = os.path.join(*path, target, featuretype, str(minTweet))
    return path

def y_categories_target_key(target):
    if 'age' in target:
        target_key = 'centered_age'
        cats = np.array([1947, 1963, 1975, 1985, 1995])

    else:
        target_key = 'gender'
        cats = np.array(['male', 'female'])  # male is dropped -> female mapped to 1

    return cats, target_key

def save_sparse(matrix:sparse.csr_matrix, path:str):
    sparse.save_npz(path, matrix)
    print('saved sparse matrix to file...')
    sys.stdout.flush()

def load_sparse(path:str):
    if '.npz' not in path:
        path = path+'.npz'
    matrix = sparse.load_npz(path)
    print('loaded sparse matrix from file...')
    sys.stdout.flush()
    return matrix

def save_uids(uids, path:str):
    with open(path, 'wb') as w:
        pickle.dump(uids, w)

def load_uids(path:str):
    with open(path, 'rb') as w:
        uids= pickle.load(w)
    return uids

def ndjson_iterator(path):

    filen = split_path_unix_win(path)[-1]
    filen = '_'.join(filen.split('_')[1:4])
    f = open(path, 'r', encoding='utf8')
    for line in f:
        try:
            line = json.loads(line)
        except:
            line = line.split('{')[1]
            line = line.split('}')[0]
            line = '{'+line+'}'
            line = json.loads(line)

        yield line[filen]

def reformat_gram_dic(gram_dict):
    order =     order = ['dist', 'char', 'emoticon_c', 'pos', 'tag', 'word', 'vect','polarity', 'num']
    reformed = {}
    for subset in ['train', 'val', 'test']:
        for featuretyp in order:
            if featuretyp in gram_dict.keys():
                for tweetLen in gram_dict[featuretyp].keys():
                    for target in gram_dict[featuretyp][tweetLen].keys():
                        for numAuth in gram_dict[featuretyp][tweetLen][subset].keys():
                            grams = []
                            #restructure dictionary
                            reformed[subset] = reformed.get(subset, {})
                            reformed[subset][target] = reformed[subset].get(target, {})
                            reformed[subset][target][tweetLen] = reformed[subset][target].get(tweetLen, {})
                            reformed[subset][target][tweetLen][numAuth] = reformed[subset][target][tweetLen].get(numAuth, {})
                            reformed[subset][target][tweetLen][numAuth][featuretyp] = reformed[subset][target][tweetLen][numAuth].get(featuretyp, {})

                            for gram in gram_dict[featuretyp][tweetLen][subset][numAuth].keys():
                                if gram == 'ids':
                                    y_path = gram_dict[featuretyp][tweetLen][subset][numAuth][gram]
                                    reformed[subset][target][tweetLen][numAuth][featuretyp]['ids'] = y_path
                                else:
                                    intGram = gram_dict[featuretyp][tweetLen][subset][numAuth][gram]['numGrams']
                                    grams.append(intGram)
                                    grams.sort()
                            reformed[subset][target][tweetLen][numAuth][featuretyp]['numGrams'] = grams

    return reformed

def get_slice_inds(vocab_dic):
    #sort list of keys type:str(numeric) because that is also the order in which we hstacked the sparse matrices
    #we need the inds for slicing
    keys = sorted(list(vocab_dic.keys()))
    slice_inds = [0]
    offset = 0
    for key in keys:
        vocab = vocab_dic[key]['vocab']
        ind = len(list(vocab.values()))
        slice_inds.append(ind+offset)
        offset +=ind

    return slice_inds

def get_slice_inds_from_feaure_inds(vocab_dic):
    keys = sorted(list(vocab_dic.keys()))
    slice_inds = [0]
    offset = 0
    for key in keys:
        upper = vocab_dic[key]['feature_inds'][1]
        slice_inds.append(upper+offset)
        offset+=upper
    return slice_inds

def combine_vocabs(vocab_dic):
    #we need a combined vocab as well as the the slice indices (to calc distortion)
    # sort list of keys type:str(numeric) because that is also the order in which we hstacked the sparse matrices
    combine = {}
    asserter = {}
    keys = sorted(list(vocab_dic.keys()))
    slice_inds = [0]
    for key in keys:
        vocab = vocab_dic[key]['vocab']
        assert asserter.get[slice_inds[-1], 0] == 0
        c=0
        for feature, item in vocab.items():
            #by default the features are unique over the vocabs (one vocab spans char-grams 1, other chargrams2 and so on)
            combine[feature] = item+slice_inds[-1]
            asserter[item] = feature
            if item > c:
                c=item

        c+=1 #this is our slicing ind
        slice_inds.append(c)
    return combine, slice_inds

def sort_weight_matrices(weights, vocabs):
    #sorts weight matrices in such a way that same feature is in same position

    weights1 = weights[0]
    weights2 = weights[1]
    vocabs1 = vocabs[0] #should be smaller - in principle so we start here
    vocabs2 = vocabs[1]
    extender2 = []
    sorter1 = []
    sorter2 = []
    keys1 = sorted(list(vocabs1.keys()))
    keys2 = sorted(list(vocabs2.keys()))

    assert len(keys1) == len(keys2) #assert num keys/vocabs
    assert sum([keys1[i] == keys2[i] for i in range(len(keys1))])== len(keys1) #assert order and same-same
    slice_inds1 = [0]
    slice_inds2 = [0]
    for key in keys2:
        t_extender2 = []
        t_sorter1 = []
        t_sorter2 = []
        vocab2 = copy.deepcopy(vocabs2[key]['vocab'])
        vocab1 = copy.deepcopy(vocabs1[key]['vocab'])
        #sort in terms of values, i.e. the indices in the weight matrix
        features = np.array(list(vocab2.keys()))
        items = np.array(list(vocab2.values()))
        sort = np.argsort(items)
        items = items[sort].tolist()
        features = features[sort].tolist()
        for i in range(0, len(features)):
            feature = features[i]
            ind2 = items[i]
            assert i == ind2 ##should now be true
            ind1 = vocab1.pop(feature, 'fail')
            if ind1 == 'fail':
                t_extender2.append(ind2 + slice_inds2[-1]) # other vocab does not have it...put it at the end
            else:
                t_sorter2.append(ind2 + slice_inds2[-1])
                t_sorter1.append(ind1 + slice_inds1[-1]) #add to sorter plus the offset (since we have many dics)
        #now we add all remaining keys in vocab1 to the end of the sorter
        for feat in vocab1.keys():
            t_sorter1.append(vocab1[feat] + slice_inds1[-1])
        t_sorter2 = t_sorter2+t_extender2
        #add new offset for indices
        slice_inds1.append(len(list(vocabs1[key].keys())))
        slice_inds2.append(len(features))

        #add to sorter and extender
        sorter1.extend(t_sorter1)
        sorter2.extend(t_sorter2)

    #now we resort both weight matrices by the respective sorter/extender
    try:
        assert weights1.shape[1] == len(sorter1)
        assert weights2.shape[1] == len(sorter2)
    except:
        print('Exception: Shape 1 is {} and len sorter is {}'.format(weights1.shape, len(sorter1)))
        print('Exception: Shape 2 is {} and len sorter is {}'.format(weights2.shape, len(sorter2)))
        sys.stdout.flush()

    weights1 = weights1[:, sorter1]
    weights2 = weights2[:, sorter2]
    weights = [weights1, weights2]

    return weights

def sort_direct_classifier_weights(weights1, weights2,
                                   vocabs1, vocabs2,
                                   inds1, inds2,
                                   featuretypes):
    #this is for a wheight matrix which is one matrix across all features
    sub1 = subslice_weights(featuretypes, inds1)
    sub2 = subslice_weights(featuretypes, inds2)
    assert len(vocabs1) == len(vocabs2)
    assert len(vocabs1[0].keys()) == len(vocabs2[0].keys())
    assert list(vocabs1[1].keys()) == list(vocabs2[1].keys())
    curr_weights = copy.deepcopy(weights2)
    for i in range(len(vocabs1)):
        dic1 = vocabs1[i]
        dic2 = vocabs2[i]
        low1 = sub1[i][0]
        upp1 = sub1[i][-1]
        low2 = sub2[i][0]
        upp2 = sub2[i][-1]
        weights = [weights1[:, low1:upp1], curr_weights[:, low2:upp2]]
        returned_weights = sort_weight_matrices(weights, [dic1, dic2])
        weights1[:, low1:upp1] = returned_weights[0]
        curr_weights[:, low2:upp2] = returned_weights[1]


    return weights1, curr_weights



def flatten_slice_inds(slice_inds):
    #makes a flat list of all indices to select in weight matrix (i.e. all those with vocabs)
    selector = []
    offset = 0
    for i, slic in enumerate(slice_inds):
        selector.extend([el+offset for el in slic if  el != 0 or i == 0])
        offset += slice_inds[i][-1]

    return selector


def subslice_weights(featuretypes, slice_inds):
    #makes a flat list of all indices to select in weight matrix (i.e. all those with vocabs)
    selector = []
    offset = 0
    for i, typ in enumerate(featuretypes):
        if not typ in ['emoticon_c''polarity', 'num']:
            selector.append([el+offset for el in slice_inds[i]])
        offset += slice_inds[i][-1]

    return selector

def train_scikit_model(model, X_train, y_train_enc, cats, batch_size=1000, epochs = 50, shuffle=True, feature='', grams=''):

    bg = generator.batch_generator(X_train, np.argmax(y_train_enc, -1).reshape((-1, 1)), batch_size=batch_size, shuffle=shuffle)
    score_holder = 0
    loss_holder = 0
    count_score = 0
    count_loss = 0
    for _ in range(epochs):
        score_curr = 0
        loss_curr = 0
        for i in range(0, int(np.ceil(X_train.shape[0] / batch_size))):
            batch = next(bg)
            if len(batch[0]) > 0:
                try:
                    model = model.partial_fit(batch[0], batch[1].flatten(), classes=np.arange(0, len(cats)))
                    #preds = model.predict(batch[0])
                    #loss_curr += model.loss_function_(batch[1].flatten(), preds)
                    score_curr += model.score(batch[0], batch[1].flatten())
                except Exception as e:
                    print('errors occured loop {} in {} {}'.format(i, feature, grams))
                    sys.stdout.flush()
                    e.args = e.args + ('\n could not make partial fit {} for feature {} and grams {}...'.format(i, feature, grams), )
                    print(e)
                    print('x contains NaN {}'.format(np.sum(np.isnan(batch[0]))))
                    print('x contains inf {}'.format(np.sum(np.isinf(batch[0]))))
                    print('y contains Nan {}'.format(np.sum(np.isnan(batch[1].flatten()))))
                    print('y contains inf {}'.format(np.sum(np.isinf(batch[1].flatten()))))
                    sys.stdout.flush()
                    score_curr =0
            else:
                print('batch was not sufficient')
                sys.stdout.flush()
        score_curr = score_curr/(i+1)
        #loss_curr = loss_curr/batch_size
        if round(score_holder,2) == round(score_curr,2):
            count_score +=1
            score_holder = score_curr
        else:
            score_holder = score_curr
            count_score = 0
        #if loss_curr > loss_holder:
        #    loss_holder = loss_curr
        #elif round(loss_holder,3) == round(loss_curr,3):
        #   count_loss +=1

        if count_loss >3 or count_score >3:
            print('\n early stopping with score {}...'.format(score_curr))
            sys.stdout.flush()
            break
        print("\n Acc score is {}".format(score_curr))
    return model

def _hotfixer(shape0, shape1):
    #hacking predictions "fixer"...fixes only the shape nothing else; preds are random bullshit
    zer = np.zeros(shape=(shape0, shape1))
    rd.seed(123456)
    if shape1 >1:
        samples = rd.sample(list(range(0, shape1)), shape0)
    else:
        samples = rd.sample([-0.5, 0.5], shape0)

    inds = list(range(0, shape0))
    zer[inds, samples] = 0.6

    return zer

def make_predictions(model, X_test, cats, batch_size=1000, shuffle=False):
    bg = generator.batch_generator(X=X_test, batch_size=batch_size, shuffle=shuffle)
    preds= []
    preds_cat = []
    for _ in range(0, int(np.ceil(X_test.shape[0] / batch_size))):
        batch = next(bg)
        if len(batch[0]) >0:
            try:
                pred_batch = model.decision_function(batch)
            except Exception as e:
                print('x contains NaN {}'.format(np.sum(np.isnan(batch[0]))))
                print('x contains inf {}'.format(np.sum(np.isinf(batch[0]))))
                print('y contains Nan {}'.format(np.sum(np.isnan(batch[1].flatten()))))
                print('y contains inf {}'.format(np.sum(np.isinf(batch[1].flatten()))))
                e.args = + '\n make predictions on batch...'
                print(e)
                sys.stdout.flush()
                shape1 = len(cats) if len(cats) >1 else 1
                pred_batch = _hotfixer(shape0=batch.shape[0], shape1=shape1)
            if len(cats)>2:
                pred_batch = local_softmax(pred_batch)
                cat_batch = [cats[el] for el in np.argmax(pred_batch, -1).flatten()]

            else:
                cat_batch = [cats[el] for el in (pred_batch > 0).astype(int).flatten()]
                #make correct shape for input into logistic
                pred_batch = pred_batch.reshape((-1,1))
            preds.append(pred_batch)
            preds_cat.extend(cat_batch)
        else:
            print('batch was not sufficient')
            sys.stdout.flush()
    preds = np.vstack(preds)
    return preds, preds_cat

def reslice_weights(weights:np.ndarray, weights_list:list):
    #takes a nd.array of weights and slices them into a list of nd.arrays according to indides given
    ind = weights_list[0].shape[1]
    out_list = [weights[:, 0:ind]]
    for i in range(1, len(weights_list)):
        #append next slice of weights
        out_list.append(weights_list[:, ind:(weights_list[i].shape[1])])
        #set new lower bound for next slice
        ind = weights_list[i].shape[1]

    return out_list

def local_softmax(vec:np.ndarray):
    if len(vec.shape) > 1:
        sm = softmax(vec, axis=1)
    else:
        sm = softmax(vec)
    return sm



