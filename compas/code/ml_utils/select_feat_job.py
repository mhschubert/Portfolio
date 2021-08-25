import os
import sys
import time
import multiprocessing
import torch
import pandas as pd

# Add code path to look for files to import
code_path = os.path.abspath('/home/llorenz/compas-analysis/code')
base_path = os.path.join(code_path, '..')
if code_path not in sys.path:
    sys.path.append(code_path)

import ml_utils
import param_search

#set datapath and dataset
data_path = os.path.join(base_path, "data", "preprocessed")
dataset = "_reduced_size"

df_X, df_y, matchframe = ml_utils.load_and_prep_data(data_path, dataset)

doubled_items = ['p_charge', 'p_famviol_arrest', "crim_inv_arrest", "crim_inv_charge", "vio_hist", "history_noncomp"]
df_X.drop(labels=doubled_items, axis=1, inplace=True)

df_X_norm = ml_utils.rescale_df(df_X, variable_dependent=True)

target = 'Risk of Violence_raw_score'
strat = None

X_train, X_val, X_test, y_train, y_val, y_test = ml_utils.custom_train_test_split(df_X_norm, df_y, matchframe,
                                                                                  dataset,
                                                                                  test_size=0.25, stratify=strat,
                                                                                  random_state=0)

#Rescale y values also
#scorelabels = [x if '_score_text' not in x for x in list(df_ys.columns.values)]
scorelabels = ['Risk of Failure to Appear_decile_score',
       'Risk of Failure to Appear_raw_score',
       'Risk of Recidivism_decile_score', 'Risk of Recidivism_raw_score',
       'Risk of Violence_decile_score', 'Risk of Violence_raw_score',
       'Risk of Recidivism_decile_score/10',
       'Risk of Violence_decile_score/10', 'target_error_regression_base',
       'target_error_regression_viol']#'recid','recid_violent',
featurelabels = list(df_X_norm.columns.values)


ys_train = ml_utils.rescale_df(y_train[scorelabels])
y_train_norm = pd.concat([ys_train, y_train.drop(scorelabels + ['Unnamed: 0'], axis=1)], axis=1)

ys_val = ml_utils.rescale_df(y_val[scorelabels])
y_val_norm = pd.concat([ys_val, y_val.drop(scorelabels + ['Unnamed: 0'], axis=1)], axis=1)

y_train_norm['group_viol'] = y_train_norm.apply(lambda x: ml_utils.marker(x['Risk of Violence_score_text'],
                                                           x.recid_violent), axis=1)

y_val_norm['group_viol'] = y_val_norm .apply(lambda x: ml_utils.marker(x['Risk of Violence_score_text'],
                                                           x.recid_violent), axis=1)

y_train_norm['group_recid'] = y_train_norm.apply(lambda x: ml_utils.marker(x['Risk of Recidivism_score_text'],
                                                           x.recid), axis=1)

y_val_norm['group_recid'] = y_val_norm .apply(lambda x: ml_utils.marker(x['Risk of Recidivism_score_text'],
                                                           x.recid), axis=1)

##make subgroups
for tp in ['group_viol', 'group_recid']:
    y_val[tp+'_FP'] =  y_val_norm[tp] == 1
    y_val[tp+'_FN'] =  y_val_norm[tp] == -1
    y_val[tp+'_TP'] =  y_val_norm[tp] == 0.5
    y_val[tp+'_TN'] =  y_val_norm[tp] == -0.5
    y_val[tp+'_T'] =  abs(y_val_norm[tp]) < 1
    y_train[tp+'_FP'] =  y_train_norm[tp] == 1
    y_train[tp+'_FN'] =  y_train_norm[tp] == -1
    y_train[tp+'_TP'] =  y_train_norm[tp] == 0.5
    y_train[tp+'_TN'] =  y_train_norm[tp] == -0.5
    y_train[tp+'_T'] =  abs(y_train_norm[tp]) < 1

target = "group_viol_FP"

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    print("Starting...")
    start = time.time()
    param_search.select_feat(base_path, X_train, X_val, y_train, y_val, target,
                             regex_keys=None,
                             n_layers=3, n_hidden=35, n_epochs=80, weighting=True,
                             batch_size=1024, activation=torch.nn.Tanh,
                             lr=0.001, weight_decay=0.001,device="cpu", n_runs=35, pool=pool)
    end = time.time()
    print("Took %.1f minutes" % ((end - start)/60))
else:
    print("Aborted.")