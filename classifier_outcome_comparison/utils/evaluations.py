import tensorflow as tf
import innvestigate
import innvestigate.utils as iutils
import numpy as np
import jsonlines
import os
import sys
import re
sys.path.append('./')
import ml_utils, generator
from sklearn.metrics import f1_score, accuracy_score
import pkg_resources
pkg_resources.require("Scipy==1.7.0")
import scipy
from scipy.stats import spearmanr

def calc_scores(y_true, y_predict):
    scores = {'accuracy': accuracy_score(y_true.flatten(), y_predict.flatten()),
              'f1-score': f1_score(y_true, y_predict, average='macro')}

    return scores

def get_shape_diff(slice_inds):
    assert len(slice_inds[0]) == len(slice_inds[1])
    slicers = []
    for i in range(1, len(slice_inds[0])):
        #iterate over endpoints of the subsets, ie. char1, char2, char3 (num features may differ for different num authors)
        diff = (slice_inds[1][i] -slice_inds[1][i-1]) - (slice_inds[0][i] -slice_inds[0][i-1])
        slicers.append({'lower': (slice_inds[0][i-1], slice_inds[0][i] ),
                 'upper': (slice_inds[1][i-1], slice_inds[1][i] ),
                 'diff': diff})
    return slicers

def make_average_distortion(weights_lower, weights_upper, slice_inds):
    #expects slice inds as a list of two slice_inds_lists; each list is a flat list of continous indices
    assert weights_lower.shape[0] == weights_upper.shape[0]

    flat_slicers = []
    flat_slicers.append(ml_utils.flatten_slice_inds(slice_inds[0]))
    flat_slicers.append(ml_utils.flatten_slice_inds(slice_inds[1]))

    slicers = get_shape_diff(flat_slicers)

    #we have to iterate over all classes (to calc distortion per class
    saver = {}
    for i in range(0, weights_lower.shape[0]):
        tmp = {}
        #we then iterate over all subfeatureparts, ie.e char1, char2, word1, word2 and calc their distortion
        for sub in range(len(slicers)):
            low = slicers[sub]['lower']
            upp = slicers[sub]['upper']

            ret = calc_distortion(weights_lower[i, (low[0]):(low[1])].reshape((1,-1)),
                                  weights_upper[i, (upp[0]):(upp[1])].reshape((1,-1)),
                                  diff=slicers[sub]['diff'])
            #add to our result
            for key in ret.keys():
                tmp[key] = tmp.get(key, 0) + ret[key]
                tmp[key+'_count'] = tmp.get(key+'_count', 0) + 1
        ##make average distortion for class
        for key in tmp.keys():
            if '_count' not in key:
                saver[key] = saver.get(key, 0) + tmp[key]/tmp[key+'_count']
                saver[key + '_count'] = saver.get(key + '_count', 0) + 1
    keys = list(saver.keys())

    #make average of average distortions
    for key in keys:
        if '_count' not in key:
            saver['Average ' + key] = saver.pop(key) / saver.pop(key + '_count')

    return saver

def calc_distortion(weights_lower, weights_upper, diff):
    #weights has shape of coeff_: (1, n_features)
    #fill up that with less features with 0 weights as those are not existent for one vector
    if diff <0:
        zero = np.zeros(shape=(weights_upper.shape[0],np.abs(diff))).flatten()
        zero = np.array([9000 for i in range(0, zero.shape[0])]).reshape((weights_upper.shape[0],np.abs(diff)))
        red = weights_lower.flatten()[:(weights_lower.shape[1] + diff)]
        other = weights_upper.flatten()
        weights_upper = np.concatenate((weights_upper, zero),axis=1)
    else:
        zero = np.zeros(shape=(weights_lower.shape[0], diff)).flatten()
        zero = np.array([9000 for i in range(0, zero.shape[0])]).reshape((weights_lower.shape[0], diff))
        red = weights_upper.flatten()[:(weights_upper.shape[1] - diff)]
        other = weights_lower.flatten()
        weights_lower = np.concatenate((weights_lower, zero),axis=1)

    # print(weights_upper.shape)
    # print(weights_lower.shape)
    # print(red.shape)
    # print(other.shape)

    #make ranks of coefficients
    ext1 = np.argsort(weights_upper.flatten(), axis=None)
    ext2 = np.argsort(weights_lower.flatten(), axis=None)

    red = np.argsort(red, axis=None).flatten()
    other = np.argsort(other, axis=None).flatten()
    spear_ext = spearmanr(a=ext1, b=ext2)
    spear_short = spearmanr(a=red, b=other)
    # print(spear_ext)
    # print(spear_short)



    return {"Spearman's Rho (ext.)":spear_ext[0], "Spearman's Rho (red.)": spear_short[0]}

def calc_updated_weights(input_weights, upper_weights, info:list, modeltype='dynAA'):
    #scikit sgd svm function is np.dot(coeff_, input) - intercept_ -> complete model up to softmax is np.dot(upp.coeff_, (np.dot(coeff_input) -intercept_))
    #slide over m axis of upper (n,m) and calc dot product for respective entry in input-weights
    info[3] = re.sub('emoticon_c', 'emoticon', info[3])
    offset = 0
    dots = []
    for el in input_weights:
        dots.append(np.dot(upper_weights[:, (0+offset):(el.shape[0]+offset)], el))
    assert upper_weights.shape[1] == sum([el.shape[0] for el in input_weights]) ##assert that number of features for upper model is number of concated inputs
    combined = np.hstack(dots)

    out_dic = {'featuretypes': info[3], '# Authors': int(info[2]), 'subgrams': info[4], 'modeltype': modeltype,
               'name': info[3].split('_')[-1] + ' ' + str(info[4][-1]), 'tweetLen': info[1], 'target': info[0],
               'weights': combined}

    return out_dic

def calc_avg_relevance_case(model:tf.keras.Model, input, info:list, modeltype='dynAA'):
    info[3] = re.sub('emoticon_c', 'emoticon', info[3])
    if model.layers[-1].output_shape >1:
        model = iutils.model_wo_softmax(model)

    analyzer = innvestigate.create_analyzer('lrp.epsilon_IB', model)
    #should not be necessary to train
    #analyzer.fit_generator(generator.batch_generator(X=input, batch_size=1000), verbose=1)
    out = np.zeros(shape=(input.shape[1], 1))
    for i in range(input.shape[0]):
        out += analyzer.analyze(input[i,:].todense()).reshape((input.shape[1], 1))
    #average relevance
    out = out/i

    out_dic = {'featuretypes':info[3], '# Authors': int(info[2]), 'subgrams':info[4], 'modeltype':modeltype,
               'name': info[3].split('_')[-1] + ' ' + str(info[4][-1]), 'tweetLen': info[1], 'target':info[0],
               'weights': out}
    return out_dic

def calc_weight_dic(weights, info:list, modeltype='direct'):
    info[3] = re.sub('emoticon_c', 'emoticon', info[3])
    out_dic = {'featuretypes':info[3], '# Authors': int(info[2]), 'subgrams':info[4], 'modeltype':modeltype,
               'name': info[3].split('_')[-1] + ' ' + str(info[4][-1]), 'tweetLen': info[1], 'target':info[0],
               'weights':weights}
    return out_dic

def make_score_table(info:list, y_true:np.ndarray, y_pred:np.ndarray, path:str, modeltype:str, mode='w'):
    sort_order = {k:v for v,k in enumerate(['dist', 'char', 'asis', 'pos', 'tag', 'dep', 'lemma', 'word', 'vect','emoticon','polarity', 'num'])}
    #fix the emoticon problem due to the "_"-separator
    info[3]= re.sub('emoticon_c', 'emoticon', info[3])
    #info = [target, tweetLen, numAuth, '_'.join(featuretypes), subgrams]
    result_dir = ml_utils.make_result_dirs(path, info[0])
    #make sort_rank for current results
    sort_rank = [sort_order[el] for el in info[3].split('_')]
    sort_rank = ''.join(str(el) for el in sort_rank)
    sort_rank = sort_rank + '0' + ''.join([str(el) for el in info[4]])

    row = {'featuretypes':info[3], '# Authors': int(info[2]), 'subgrams':info[4], 'modeltype':modeltype,
           'name': info[3].split('_')[-1] + ' ' + str(info[4][-1]), 'tweetLen': info[1], 'target':info[0], 'rank':sort_rank}

    row.update(calc_scores(y_true, y_pred))


    if mode != 'return':
        with open(os.path.join(result_dir, 'scores.ndjson') , mode) as w:
            writer = jsonlines.Writer(w)
            writer.write(row)
            writer.close()

        print('wrote scores to file')
        sys.stdout.flush()
    else:
        return row

def make_distortion_table(info:list, lower_dic:dict, upper_dic:dict, slice_inds:list , path:str, modeltype:str, mode='w'):
    sort_order = {k:v for v,k in enumerate(['dist', 'char', 'asis', 'pos', 'tag', 'dep', 'lemma', 'word', 'vect','emoticon','polarity', 'num'])}
    #fix the emoticon problem due to the "_"-separator
    info[3]= re.sub('emoticon_c', 'emoticon', info[3])
    #info = [target, tweetLen, numAuth, '_'.join(featuretypes), subgrams]
    result_dir = ml_utils.make_result_dirs(path, info[0])
    #make sort_rank for current results
    sort_rank = [sort_order[el] for el in info[3].split('_')]
    sort_rank = ''.join(str(el) for el in sort_rank+info[4])

    row = {'featuretypes':info[3], '# Authors': int(info[2]), 'subgrams':info[4], 'modeltype':modeltype,
           'name': info[3].split('_')[-1] + ' ' + str(info[4][-1]), 'tweetLen': info[1], 'target':info[0], 'rank':sort_rank}

    assert lower_dic['modeltype'] == upper_dic['modeltype']

    if lower_dic['modeltype'] == 'dynAA':
        row.update(make_average_distortion(lower_dic['weights'], upper_dic['weights'],
                                           slice_inds=slice_inds))
    else:

        row.update(make_average_distortion(lower_dic['weights'], upper_dic['weights'], slice_inds=slice_inds))
    if mode != 'return':
        print(mode)
        with open(os.path.join(result_dir, 'distortion.ndjson'), mode) as w:
            writer = jsonlines.Writer(w)
            writer.write(row)
            writer.close()

        print('wrote distortion to file')
        sys.stdout.flush()
    else:
        return row


