import os
import sys
import pandas as pd
import numpy as np
import scipy as sci
import math
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import preprocessing
import imblearn.over_sampling as ib
from sklearn.metrics import accuracy_score
# Plot style
sns.set(style="whitegrid")




def visualize_relevance(labels, relevance, target_label):
    # Visualize feature-wise relevance scores as barchart
    plt.figure(figsize=(8, 8))
    # print([(labels[i], relevance[i]) for i in range(len(labels))])
    plt.barh(labels, relevance, height=1)
    plt.title(target_label)
    plt.xlabel("Relevance")
    plt.show()

def _get_model_name_dict():
    dic = {
        'Risk of Violence_decile_score/10': 'viol_dec10',
        'Risk of Violence_decile_score': 'viol_dec',
        'Risk of Violence_raw_score': 'viol_raw',
        'Risk of Recidivism_decile_score/10': 'rec_dec10',
        'Risk of Recidivism_decile_score': 'rec_dec',
        'Risk of Recidivism_raw_score': 'rec_raw',
    }
    return dic


def save_model(model, base_path, name):
    try:
        name = _get_model_name_dict()[name]
    except:
        pass
    print("Saving %s model." % name)

    path = os.path.join(base_path, "saves")
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, name + ".model"))


def load_model(name, base_path):
    try:
        name = _get_model_name_dict()[name]
    except:
        pass
    print("Loading %s model." % name)

    path = os.path.join(base_path, "saves")
    model = torch.load(os.path.join(path, name + ".model"))
    return model


def make_weight_table(labels):
    
    labels = list(labels)
    labels = [float(l) for l in labels]
    c = Counter(labels)
    l = len(labels)
    frac = {}
    for key, count in c.items():
        frac[key] = 1/(count/l)
    
    del c
    
    return frac


def oversample(X_train, y_train, target, weighting= ['random'], random_state = 12345):
    matching_dic = {'Risk of Recidivism_decile_score': 'Risk of Recidivism_decile_score',
              'Risk of Recidivism_decile_score/10': 'Risk of Recidivism_decile_score',
              'Risk of Recidivism_raw_score': 'Risk of Recidivism_decile_score',
              'recid': 'recid',
              'recid_proPub': 'recid_proPub',
              'Recidivism_decile': 'Recidivism_decile',
              'Recidivism_decile/10': 'Recidivism_decile',
              'Risk of Violence_decile_score': 'Risk of Violence_decile_score',
              'Risk of Violence_decile_score/10': 'Risk of Violence_decile_score',
              'Risk of Violence_raw_score': 'Risk of Violence_decile_score',
              'recid_violent': 'recid_violent',
              'recid_violent_proPub': 'recid_violent_proPub',
             }
 
    cat_mask = ['p_current_on_probation', 'is_misdem','race_black', 'race_white', 'race_hispanic', 'race_asian',
                'race_native', 'is_male']
    weighting += [5,5]
    colnames = list(X_train.columns)
    #x_index = X_train.index.values
    #y_index = y_train.index.values
    cat_mask = np.array([True if col in cat_mask else False for col in colnames ] + [True])
    
    oversampler = {'random': ib.RandomOverSampler(sampling_strategy='auto', random_state=random_state),
                  'ADASYN': ib.ADASYN(sampling_strategy='not majority', random_state=random_state,
                                      n_neighbors=weighting[1], n_jobs=1),
                  'SMOTE': ib.SMOTE(sampling_strategy='not majority', random_state=random_state, k_neighbors=weighting[1]),
                  'SMOTENC': ib.SMOTENC(categorical_features=cat_mask, sampling_strategy='not majority',
                                        random_state=random_state,k_neighbors=weighting[1], n_jobs=1),
                  'BorderlineSMOTE': ib.BorderlineSMOTE(sampling_strategy='not majority', random_state=random_state,
                                                        k_neighbors=weighting[1],
                                                        n_jobs=1,m_neighbors=10, kind='borderline-1'),
                   'SVMSMOTE': ib.SVMSMOTE(sampling_strategy='not majority', random_state=random_state,
                                           k_neighbors=weighting[1], n_jobs=1,
                                           m_neighbors=weighting[2], svm_estimator=None, out_step=0.5),
                   'KMeansSMOTE':ib.KMeansSMOTE(sampling_strategy='not majority', random_state=random_state, 
                                                k_neighbors=weighting[1], n_jobs=1, kmeans_estimator=None,
                                                cluster_balance_threshold='auto',
                                                density_exponent='auto'),
                  }
    
    strategy = weighting[0]
  
    X_train['target'] = y_train[target]
    X_os, y_os = oversampler[strategy].fit_resample(X_train, y_train[matching_dic[target]])
    X_train.drop(labels=['target'], inplace = True, axis=1)
    if strategy == 'random':
               
        i_os = list(oversampler[strategy].sample_indices_)
        
        y_os = y_train.iloc[i_os]
        X_os = X_train.iloc[i_os]
    else:
        #label score - mostly decile
        y_os = np.array([list(y_os)])
        #adding the actual target score
        y_os = np.transpose(np.concatenate((y_os[0:,], np.array([list(X_os['target'])])), axis=0))

        y_os = pd.DataFrame(data= y_os[0:,0:],
                            index = [i for i in range(y_os.shape[0])],
                            columns = [matching_dic[target], target])

        #remove generated target score from x
        X_os.drop(labels=['target'], inplace=True, axis=1)

    return X_os, y_os


def rescale_df(df, min_val=0, max_val=1, variable_dependent=False):
    # Scale data to [0,1]
    min_max_scaler = preprocessing.MinMaxScaler((min_val, max_val))
    np_scaled = min_max_scaler.fit_transform(df)
    df_X_norm = pd.DataFrame(np_scaled, columns=df.columns, index=df.index)

    if variable_dependent:
        for c in df_X_norm.columns:
            if str(c) in ["p_current_age", "p_age_first_offense"]:
                # min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
                min_max_scaler = preprocessing.StandardScaler()
                np_scaled = min_max_scaler.fit_transform(pd.DataFrame(df[c]))
                # print(np_scaled)
                df_X_norm[c] = np_scaled

    return df_X_norm

def custom_train_test_split(df_X_norm, df_y, matchframe, dataset, test_size=0.25, stratify=None, random_state=0):
    if "reduced_size" not in dataset:

        df_X_s = df_X_norm.loc[list(matchframe.is_reduced)]
        df_y_s = df_y.loc[list(matchframe.is_reduced)]
        df_X_e = df_X_norm.loc[[not i for i in list(matchframe.is_reduced)]]
        df_y_e = df_y.loc[[not i for i in list(matchframe.is_reduced)]]
        if stratify:
            stratl = df_y_s[stratify]
        else:
            stratl = None
        X_train, X_val, y_train, y_val = train_test_split(df_X_s, df_y_s, test_size=test_size,
                                                          random_state=random_state, stratify=stratl)

        if stratify:
            stratl = y_train[stratify]
            # split into train and validation; TODO possibly stratify
        X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=test_size,
                                                        random_state=random_state, stratify=stratl)

        ##add those inds to train set which are not in the reduced set
        X_train = pd.concat([X_train, df_X_e], axis=0)
        y_train = pd.concat([y_train, df_y_e], axis=0)



    else:
        if stratify:
            stratl = df_y[stratify]
        else:
            stratl = None
        X_train, X_test, y_train, y_test = train_test_split(df_X_norm, df_y, test_size=test_size,
                                                            random_state=random_state, stratify=stratl)
        if stratify:
            stratl = y_train[stratify]
        else:
            stratl = None
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size,
                                                          random_state=random_state, stratify=stratl)

    return X_train, X_val, X_test, y_train, y_val, y_test

def marker(cat, recid, group = ['medium','high']):
    
    if cat.lower() in group and recid == 1:
        return(0.5)
    elif cat.lower() in group and recid !=1:
        return(1)
    elif cat.lower() not in group and recid == 1:
        return(-1)
    elif cat.lower() not in group and recid !=1:
        return(-0.5)
    

def iterative_marker(dec, recid, iterative_cutoff=4):
    if dec >= iterative_cutoff and recid ==1:
        return(0.5)
    elif dec >= iterative_cutoff and recid !=1:
        return(1)
    elif dec < iterative_cutoff and recid == 1:
        return(-1)
    elif dec < iterative_cutoff and recid !=1:
        return(-0.5)


def calc_rates(rates_list, compas_pred, true_recid, cutoff, abso=False):
    tp = [1 if rates_list[i] < 1 and compas_pred[i] >= cutoff and true_recid[i] == 1 else 0 for i in
          range(len(rates_list))]
    fp = [1 if rates_list[i] < 1 and compas_pred[i] >= cutoff and true_recid[i] != 1 else 0 for i in
          range(len(rates_list))]
    tn = [1 if (rates_list[i] >= 1 and compas_pred[i] >= cutoff and true_recid[i] != 1) or (
                compas_pred[i] < cutoff and true_recid[i] != 1) else 0 for i in range(len(rates_list))]
    fn = [1 if (rates_list[i] >= 1 and compas_pred[i] >= cutoff and true_recid[i] == 1) or (
                compas_pred[i] < cutoff and true_recid[i] == 1) else 0 for i in range(len(rates_list))]

    try:
        fd = sum(fp)/sum(fp+tp)
    except:
        fd = 0
    try:
        fo = sum(fn)/sum(fn+tn)
    except:
        fo = 0
    try:
        omr = sum(fp+fn)/sum(tp+fp+tn+fn)
    except:
        omr = 0
    if abso:
        return sum(tp), sum(fp), sum(tn), sum(fn), fd, fo, omr

    if sum(true_recid) > 0:
        tp = sum(tp) / sum(true_recid)
    else:
        tp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        fp = sum(fp) / (len(true_recid) - sum(true_recid))
    else:
        fp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        tn = sum(tn) / (len(true_recid) - sum(true_recid))

    else:
        tn = 0
    if sum(true_recid) > 0:
        fn = sum(fn) / sum(true_recid)

    else:
        fn = 0

    return tp, fp, tn, fn, fd, fo, omr


def calc_rates_two_models(rates_list, true_recid, abso=False):
    tp = [1 if rates_list[i] == 1 and true_recid[i] == 1 else 0 for i in range(len(rates_list))]
    fp = [1 if rates_list[i] == 1 and true_recid[i] != 1 else 0 for i in range(len(rates_list))]
    tn = [1 if rates_list[i] != 1 and true_recid[i] != 1 else 0 for i in range(len(rates_list))]
    fn = [1 if rates_list[i] != 1 and true_recid[i] == 1 else 0 for i in range(len(rates_list))]

    try:
        fd = sum(fp) / sum(fp + tp)
    except:
        fd = 0
    try:
        fo = sum(fn) / sum(fn + tn)
    except:
        fo = 0

    try:
        omr = sum(fp + fn) / sum(tp + fp + tn + fn)
    except:
        omr = 0

    if abso:
        return sum(tp), sum(fp), sum(tn), sum(fn), fd, fo, omr

    if sum(true_recid) > 0:
        tp = sum(tp) / sum(true_recid)
    else:
        tp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        fp = sum(fp) / (len(true_recid) - sum(true_recid))
    else:
        fp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        tn = sum(tn) / (len(true_recid) - sum(true_recid))

    else:
        tn = 0
    if sum(true_recid) > 0:
        fn = sum(fn) / sum(true_recid)

    else:
        fn = 0

    return tp, fp, tn, fn, fd, fo, omr



def calc_compas_proportions(compas_pred, true_recid, cutoff, abso=False):


    tp = [1 if compas_pred[i] >= cutoff and true_recid[i] == 1 else 0 for i in range(len(compas_pred))]
    fp = [1 if compas_pred[i] >= cutoff and true_recid[i] != 1 else 0 for i in range(len(compas_pred))]
    tn = [1 if compas_pred[i] < cutoff and true_recid[i] != 1 else 0 for i in range(len(compas_pred))]
    fn = [1 if compas_pred[i] < cutoff and true_recid[i] == 1 else 0 for i in range(len(compas_pred))]

    try:
        fd = sum(fp) / sum(fp + tp)
    except:
        fd = 0
    try:
        fo = sum(fn) / sum(fn + tn)
    except:
        fo = 0
    try:
        omr = sum(fp + fn) / sum(tp + fp + tn + fn)
    except:
        omr = 0

    c = tp + tn
    if abso:
        return sum(tp), sum(fp), sum(tn), sum(fn), sum(c), fd, fo, omr

    if sum(true_recid) > 0:
        tp = sum(tp) / sum(true_recid)
    else:
        tp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        fp = sum(fp) / (len(true_recid) - sum(true_recid))
    else:
        fp = 0
    if (len(true_recid) - sum(true_recid)) > 0:
        tn = sum(tn) / (len(true_recid) - sum(true_recid))

    else:
        tn = 0
    if sum(true_recid) > 0:
        fn = sum(fn) / sum(true_recid)

    else:
        fn = 0


    return tp, fp, tn, fn, round(sum(c)/len(c), 2), fd, fo, omr


def use_compas_pred_for_missing(y_val, idc, pred, cutoff, col):
    tmp = y_val.copy(deep=True)
    # get keys (pos and negative)

    for key in idc.keys():
        if '_FP' in str(key):
            posk = str(key)
        else:
            negk = str(key)
    posi = [elem for elem in list(idc[posk][cutoff[0]])]
    negi = [elem for elem in list(idc[negk][cutoff[0]])]

    conc = [posi[i] or negi[i] for i in range(len(posi))]
    n_idx = [not elem for elem in [posi[i] or negi[i] for i in range(len(posi))]]

    ##add the COMPAS cutoff jailing, if the individ. does not appear in FP or FN list
    # print('Amount of positive indices for cutoff {}'.format(cutoff[1]))
    # print(Counter(posi))
    # print('Amount of negative indices for cutoff {}'.format(cutoff[1]))
    # print(Counter(negi))
    # print('{} FP predictions of DT for cutoff {}'.format(sum(pred[posk][cutoff[0]]), cutoff[1]))
    # print('{} FN predictions of DT for cutoff {}'.format(sum(pred[negk][cutoff[0]]), cutoff[1]))

    # with two models there should be nothing happening here
    tmp.loc[n_idx, 'pred'] = [1 if el >= cutoff[1] else 0 for el in tmp.loc[n_idx, col]]

    # only add predictions if there are acutally predictions made
    if True in posi:
        # invert FP as it would mean that DT assumes them to be negative
        tmp.loc[posi, 'pred'] = [abs(el - 1) for el in pred[posk][cutoff[0]]]
    if True in negi:
        tmp.loc[negi, 'pred'] = pred[negk][cutoff[0]]

    return list(tmp.pred)


##fix calculation and divsion
def calc_prorportions(y_val, rates_list_l, bool_select, compas_pred, true_recid,
                      rn=range(1, 10), mask=None, abso=False, zafar=False, index_gini = False):
    if type(mask) == type(None):
        mask = [True for _ in range(0, y_val.shape[0])]
    tp_p = []
    fp_p = []
    tn_p = []
    fn_p = []
    fd_p = []
    fo_p = []
    omr_p = []
    tp_c = []
    fp_c = []
    tn_c = []
    fn_c = []
    fd_c = []
    fo_c = []
    omr_c = []
    corr = []
    corr_c = []
    gini = []
    gini_c = []

    #Zafar et al. 2017 , false discovery, false omission, overall misclassification rate

    for i, r in enumerate(rn):

        if index_gini:
            tp_t, fp_t, tn_t, fn_t, cc, fd, fo, omr = calc_compas_proportions(y_val.loc[mask, compas_pred],
                                                                              y_val.loc[mask, true_recid], r, abso=True)

            gini_c.append(gini_index_from_contingency(tp_t, fp_t, tn_t, fn_t))

        else:

            tp_t, fp_t, tn_t, fn_t, cc, fd, fo, omr = calc_compas_proportions(y_val.loc[mask, compas_pred],
                                                             y_val.loc[mask, true_recid], r, abso=abso)



        tp_c.append(tp_t)
        fp_c.append(fp_t)
        tn_c.append(tn_t)
        fn_c.append(fn_t)
        fd_c.append(fd)
        fo_c.append(fo)
        omr_c.append(omr)
        corr_c.append(cc)
        pred = use_compas_pred_for_missing(y_val, bool_select, rates_list_l, (i, r), compas_pred)

        #calc number of correct predictions
        c = np.sum(y_val.loc[mask, true_recid].to_numpy() == np.array(pred)[mask])
        if not abso:
            c = c/y_val.loc[mask, true_recid].shape[0]
        corr.append(c)

        # print(Counter(pred))
        # print('\n')
        # tp, fp, tn, fn = calc_rates(list(np.array(pred)[mask]), y_val.loc[mask, compas_pred],
        #                            y_val.loc[mask, true_recid], cutoff=r, abso = abso)

        if index_gini:
            tp, fp, tn, fn, fd, fo, omr = calc_rates_two_models(list(np.array(pred)[mask]),
                                               y_val.loc[mask, true_recid], abso=True)

            gini.append(gini_index_from_contingency(tp, fp, tn, fn))
        else:
            tp, fp, tn, fn, fd, fo, omr = calc_rates_two_models(list(np.array(pred)[mask]),
                                               y_val.loc[mask, true_recid], abso=abso)
        tp_p.append(tp)
        fp_p.append(fp)
        tn_p.append(tn)
        fn_p.append(fn)
        fd_p.append(fd)
        fo_p.append(fo)
        omr_p.append(omr)

    if index_gini:
        return gini, gini_c
    elif zafar:
        return fd_p, fp_p, fo_p, fn_p, fd_c, fp_c, fo_c, fn_c, corr, corr_c
    else:
        return tp_p, fp_p, tn_p, fn_p, tp_c, fp_c, tn_c, fn_c, corr, corr_c



def transform_model_results(results_dic, result_key, mask = None, abso=False, zafar=False):

    tpc = []
    fpc = []
    fnc = []
    tnc = []
    tpm = []
    fpm = []
    fnm = []
    tnm = []
    corrm = []
    corrc = []

    y_true = np.array(results_dic[result_key][0]['true']).flatten()
    if type(mask) == type(None):
        mask = [True for _ in range(0, y_true.shape[0])]
    n_divisorm = 1
    p_divisorm = 1
    n_divisorc = 1
    p_divisorc = 1
    fractor = 1


    for i in results_dic[result_key].keys():
        preds = np.array([1 if el >=0.5 else 0 for el in results_dic[result_key][i]['y_hat']])
        compas = np.array(results_dic[result_key][i]['compas']).flatten()
        corrected = np.abs(compas-preds)
        if not abso or zafar:
            ## TP + FN
            n_divisorm = np.sum((corrected[mask] == 1) & (corrected[mask] == y_true[mask])) + np.sum(corrected[mask] < y_true[mask])
            ## TN + FP
            p_divisorm = np.sum((corrected[mask] == 0) & (corrected[mask] == y_true[mask])) +np.sum(corrected[mask] > y_true[mask])

            n_divisorc = np.sum((compas[mask] == 1) & (compas[mask] == y_true[mask])) + np.sum(compas[mask] < y_true[mask])
            p_divisorc = np.sum((compas[mask] == 0) & (compas[mask] == y_true[mask])) +  np.sum(compas[mask] > y_true[mask])
            fractor = y_true.shape[0]

        ##false positive rate FPR = FP/N = FP/(FP+TN)
        fpm.append(np.sum(corrected[mask] > y_true[mask]) / p_divisorm)
        ##true negative rate FNR = FN/P = FN/(FN+TP)
        fnm.append(np.sum(corrected[mask] < y_true[mask]) / n_divisorm)

        fpc.append(np.sum(compas[mask] > y_true[mask]) / p_divisorc)
        fnc.append(np.sum(compas[mask] < y_true[mask]) / n_divisorc)
        corrm.append(np.sum(corrected==y_true)/fractor)
        corrc.append(np.sum(compas==y_true)/fractor)
        if not zafar:

            # true positive rate TPR = TP/P = TP/(TP+FN)
            tpm.append(np.sum((corrected[mask] == np.array(1)) & (y_true[mask] == np.array(1)))/n_divisorm)
            #true negative rate TPN = TN/N = TN/(TN+FP)
            tnm.append(np.sum((corrected[mask] == np.array(0)) & (y_true[mask] == np.array(0)))/p_divisorm)

            tpc.append(np.sum((compas[mask] == np.array(1)) & (y_true[mask] == np.array(1)))/n_divisorm)
            tnc.append(np.sum((compas[mask] == np.array(0)) & (y_true[mask] == np.array(0)))/p_divisorm)

        else:

            ##False Discovery FDR = FP/PP = FP/(TP+FP)
            tpm.append(np.sum(corrected[mask] > y_true[mask] )/(np.sum(corrected[mask] > y_true[mask]) + np.sum((corrected[mask] == 1) & (corrected[mask] == y_true[mask]))))

            ##False Omission FOR = FN/PN = FN/(FN+TN)
            tnm.append(np.sum(corrected[mask] < y_true[mask])/(np.sum(corrected[mask] < y_true[mask]) + np.sum((corrected[mask] == 0) & (corrected[mask] == y_true[mask]))))

            tpc.append(np.sum(compas[mask] > y_true[mask] )/(np.sum(compas[mask] > y_true[mask]) + np.sum((compas[mask] == 1) & (compas[mask] == y_true[mask]))))
            tnc.append(np.sum(compas[mask] < y_true[mask])/(np.sum(compas[mask] < y_true[mask]) + np.sum((compas[mask] == 0) & (compas[mask] == y_true[mask]))))





    return tpm, fpm, tnm, fnm, tpc, fpc, tnc, fnc, corrm, corrc




def bracketing_age(pds_age):
    ret = []
    for i in list(pds_age):
        if i <= 21:
            ret.append(str(0))
        elif i <= 30:
            ret.append(str(1))
        elif i <= 40:
            ret.append(str(2))
        elif i <= 50:
            ret.append(str(3))
        elif i <= 65:
            ret.append(str(4))
        elif i > 65:
            ret.append(str(5))
        else:
            raise SystemExit('value error')

    return ret


def gini_index_from_contingency(tp, fp, tn, fn):

    if type(tp) == type(list):
        total = len(tp + fp + tn + fn)
        pos = len(tp+fp)
        neg = len(fn+tn)
    else:
        total = sum([tp, fp, tn, fn])
        pos = sum([tp, fp])
        neg = sum([tn, fn])

    weight_p = len(tp+fp)/total
    weight_n = len(tn+fn)/total
    return (1- (fp/pos)**2 - (tp/pos)**2)*weight_p + (1-(fn/neg)**2 - (tn/neg)**2)*weight_n


def get_raw(perc):

    return np.log(perc/(1-perc))



def get_percentages(raw_scores, typ='logit'):

    return np.exp(raw_scores)/(np.exp(raw_scores)+1)

    np.log(np.exp(raw_scores)) - np.log(np.exp(raw_scores))


def get_cutoffs(raw_sores, deciles):

    raw_sores = np.array(raw_sores)
    deciles = np.array(deciles)
    decile_raw = {}

    for decile in np.unique(deciles):
        mask = deciles == decile
        un = np.unique(raw_sores[mask])
        maxi = np.max(un)
        decile_raw[decile] = (maxi, round(get_percentages(maxi),3)
                              )

    return decile_raw


def exp_loss_discrete_estimate(distrib, cutoff_low=0, cutoff_high=10, decile_raw = None, is_percentage = False):
    divid = 0
    divis = 0
    if decile_raw:
        if cutoff_high != 10:
            cutoff_high = decile_raw[cutoff_high][0]
        else:
            cutoff_high = 10
        if cutoff_low != 10:
            cutoff_low = decile_raw[cutoff_low][0]
        else:
            cutoff_low = 10
    else:
        raise NotImplementedError('You have to provide a cutoff mapping')

    c = Counter(np.round(distrib, 2))
    for key, item in c.items():
        if key < cutoff_high and key >= cutoff_low:
            perc = key
            if not is_percentage:
                perc = get_percentages(key)
            divid += (perc*item)
        divis += item
    return divid/divis

def exp_loss_inverse_discrete_estimate(distrib, cutoff_low=0, cutoff_high=10, decile_raw = None, is_percentage = False):
    divid = 0
    divis = 0
    if decile_raw:
        if cutoff_high != 10:
            cutoff_high = decile_raw[cutoff_high][0]
        else:
            cutoff_high = 10

        if cutoff_low != 10:
            cutoff_low = decile_raw[cutoff_low][0]
        else:
            cutoff_low = 10
    else:
        raise NotImplementedError('You have to provide a cutoff mapping')

    c = Counter(np.round(distrib, 2))
    for key, item in c.items():
        if key < cutoff_high and key >= cutoff_low:
            perc = 1-key
            if not is_percentage:
                perc = 1-get_percentages(key)
            divid += (perc*item)
        divis += item
    return divid/divis


def exp_loss(cutoff_low, cutoff_high, distrib, decile_raw = None, is_percentage = False):
    if decile_raw:
        cutoff_high = decile_raw[cutoff_high][0]
        cutoff_low = decile_raw[cutoff_low][0]
    else:
        raise NotImplementedError('You have to provide a cutoff mapping')

    if not is_percentage:

        perc_low = get_percentages(cutoff_low)
        perc_high = get_percentages(cutoff_high)

        perc_values = get_percentages(distrib)

    kde = sci.stats.gaussian_kde(perc_values)
    #kde = sci.stats.norm(loc=0.5, scale=0.8)

    #res = sci.integrate.quad(lambda x: kde(x), 0.0, perc)[0]/sci.integrate.quad(lambda x: kde(x), 0.0, 1.0)[0]
    res = sci.integrate.quad(lambda x: kde(x), perc_low, perc_high)[0]

    return res



def calc_errors(pred, compas, y_true):
    # tensor has shape (samples, class) [0 or 1] Positive or Negative
    pred = torch.tensor(pred).reshape((-1,1))
    y_true = y_true.reshape((-1, 1))
    compas = compas.reshape((-1, 1))
    errors = torch.tensor(y_true[:, 0] != compas[:, 0]).reshape(
        (-1, 1))  ##find false positives and false negatives
    error_not_found = torch.lt(pred, 0.5) & errors

    err_falsely_found = torch.ge(pred, 0.5).float()*(~errors).float()  # this mask finds those where compas did not make an error but our model thought it is an error


    FN = torch.sum(error_not_found*torch.eq(y_true, 1).float() + err_falsely_found*torch.eq(y_true, 1).float())
    FP = torch.sum(error_not_found*torch.eq(y_true, 0).float() + err_falsely_found*torch.eq(y_true, 0).float())

    return FN, FP


def calc_accuracy(pred, compas, y_true):
    pred = [1 if float(el) >= .5 else 0 for el in pred]
    compas = [el for el in compas]
    corrected = [float(np.abs(compas[i] - pred[i])) for i in range(len(pred))]

    return accuracy_score(corrected, y_true)


def total_correct(pred, compas, y_true):
    pred = [1 if float(el) >= .5 else 0 for el in pred]
    compas = [el for el in compas]
    corrected = [float(np.abs(compas[i] - pred[i])) for i in range(len(pred))]
    return torch.sum(torch.eq(torch.tensor(corrected), torch.tensor(y_true)).float())

def calc_error_accuracy(pred, compas, y_true):
    pred = np.array([1 if float(el) >= .5 else 0 for el in pred])

    errors = np.array(compas != y_true).flatten().astype('float')
    print('Correctly marked errors: {}'.format(np.sum((errors == 1) & (errors == pred))))
    print('Incorrectly marked errors: {}'.format(np.sum((errors == 0) & (errors != pred))))
    return accuracy_score(pred, errors)



def calc_fp_fn_rates(diff, ground_truth, errors ):
    pred =errors.float() - diff
    err_not_found = torch.lt(pred, 0.5).float() *errors.float()
    err_found = torch.ge(pred, 0.5).float() * errors.float()
    inv_ground = 1-ground_truth
    err_falsely_found = (~errors).float()*torch.ge(pred, 0.5).float()


    FN = torch.mean(err_not_found*ground_truth + err_falsely_found*ground_truth)
    FP = torch.mean(err_not_found*inv_ground + err_falsely_found*inv_ground)

    return FN, FP


def calc_y_hats_dic(y_hats, y_pred, key, cutoff, y, expected):
    y_pred_t = [float(torch.sigmoid(a)) for a in y_pred]
    y_hats[key][cutoff] = {}
    y_hats[key][cutoff]['expeted_rates'] = expected
    # y_hats[key][cutoff]['model'] = returner["model"]
    y_hats[key][cutoff]['y_hat'] = torch.tensor(y_pred_t)
    y_hats[key][cutoff]['compas'] = y['group_recid_P']
    y_hats[key][cutoff]['true'] = y['recid']
    y_hats[key][cutoff]['compas_acc'] = accuracy_score(y['recid'], y['group_recid_P']) # 'group_recid_P' equals compas predictions interacted with cutoff
    y_hats[key][cutoff]['compas_fp'] = np.sum(y['recid'] < y['group_recid_P'])
    y_hats[key][cutoff]['compas_fn'] = np.sum(y['recid'] > y['group_recid_P'])
    # y_sanity = torch.tensor([0 for el in y_hats[key][cutoff]['y_hat']])
    y_hats[key][cutoff]['model_fp'], y_hats[key][cutoff]['model_fn'] = calc_errors(
        y_hats[key][cutoff]['y_hat'],
        torch.tensor(y['recid']),
        torch.tensor(y['group_recid_P']))

    acc = calc_accuracy(y_hats[key][cutoff]['y_hat'], y['group_recid_P'], y['recid'])
    print(acc)
    print('Number of COMPAS correct ones {}'.format(np.sum(y['recid'] == y['group_recid_P'])))
    print('Total number of correct ones: {}'.format(
        total_correct(y_hats[key][cutoff]['y_hat'], y['group_recid_P'], y['recid'])))
    y_hats[key][cutoff]['model_acc'] = acc

    return y_hats
