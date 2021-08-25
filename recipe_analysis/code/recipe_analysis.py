#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import os, sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV,GridSearchCV
import timeit
from tqdm import tqdm


# Don't truncate column lists
pd.options.display.max_columns = None

# Add code path to look for files to import
code_path = os.path.abspath('support')
base_path = os.path.join(code_path, '..')
if code_path not in sys.path:
    sys.path.append(code_path)
    
import ml_utils

#import warnings
#warnings.filterwarnings('ignore')

#variable setting whether precomputed data should be recomputed
#(increases runtime by 2 magnitudes)
rerun = False




if os.path.exists('../data/preprocessed/X_nbr.json') and not rerun:
    X_nbr = pd.read_json('../data/preprocessed/X_nbr.json')
    y_nbr = pd.read_json('../data/preprocessed/y_nbr.json')
    X_nbr= X_nbr.sort_values(by=['ID_recipe'])
    y_nbr = y_nbr.sort_values(by=['ID'])
else:
    print('load_fail')
    sys.exit()


# Now we employ some dimensionality reduction. Then we will feed it to some clustering algorithm in order to see the topology of our data.

# In[3]:


ft = ml_utils.Featurizer(idx='ID_recipe',
                        vectorize='ingredients',
                        feat_not_vectorize=['num_ingr'],
                        min_df = 20)
X_one_hot = ft.fit_transform(X_nbr)


# In[6]:


ID_train, ID_test = train_test_split(np.array(X_nbr.ID_recipe), train_size=0.7,
                                     random_state=1234567,
                                     shuffle=True,
                                     stratify=y_nbr.cuisine)

ID_test, ID_val = train_test_split(ID_test, train_size=0.6666666,
                                     random_state=1234567,
                                     shuffle=True,
                                     stratify=y_nbr.cuisine.loc[y_nbr.ID.isin(ID_test)])


# In[6]:


##train, test, val
X_train = X_one_hot.loc[X_one_hot.index.isin(ID_train), :].copy(deep=True)
X_test = X_one_hot.loc[X_one_hot.index.isin(ID_test), :].copy(deep=True)
X_val = X_one_hot.loc[X_one_hot.index.isin(ID_val), :].copy(deep=True)

y_train = y_nbr.loc[y_nbr.ID.isin(ID_train),:].copy(deep=True)
y_test = y_nbr.loc[y_nbr.ID.isin(ID_test),:].copy(deep=True)
y_val = y_nbr.loc[y_nbr.ID.isin(ID_val),:].copy(deep=True)


##make TDIDF
tfidf = ml_utils.Featurizer(idx='ID_recipe',
                        vectorize='ingredients',
                        feat_not_vectorize=['num_ingr'],
                        min_df = 20,
                           tfidf=True)
X_tfidf_train = tfidf.fit_transform(X_nbr.loc[X_nbr.ID_recipe.isin(ID_train),:])
X_tfidf_test = tfidf.fit_transform(X_nbr.loc[X_nbr.ID_recipe.isin(ID_test),:])
X_tfidf_val = tfidf.fit_transform(X_nbr.loc[X_nbr.ID_recipe.isin(ID_val),:])


# In[7]:


cf = ml_utils.Cuisine_Feat_Creator(selectcol='cuisine', proportional = True,
                                  not_part=X_train.shape[1])
cf.fit(X_train, y_train)
X_train = cf.transform(X_train)
X_test = cf.transform(X_test)
X_test = cf.transform(X_val)

idcol = int(X_train.columns.get_loc("num_ingr"))+1
X_tfidf_train = pd.concat([X_tfidf_train, X_train.iloc[:,idcol:].copy(deep=True)],
                          axis=1).reindex(X_tfidf_train.index)
X_tfidf_test  = pd.concat([X_tfidf_test,X_test.iloc[:,idcol:].copy(deep=True)],
                          axis=1).reindex(X_tfidf_test.index)
X_tfidf_val  = pd.concat([X_tfidf_val,X_val.iloc[:,idcol:].copy(deep=True)],
                         axis=1).reindex(X_tfidf_val.index)


# In[9]:

estimators = {'linSVM': [('linSVM', LinearSVC(random_state=1234567))],
              'NystroemSvm': [('nystroem', Nystroem(random_state=1234567)),
                              ('svm', LinearSVC(random_state=1234567,
                                               dual = False))
                             ],
              
              'logistic': [('logistic', LogisticRegression(random_state=1234567,
                                                          max_iter=5000,
                                                          solver='saga'))]

            }

parameter_dic = {
    'linSVM':[{'linSVM__C': np.logspace(-3, 3, 8),
             'linSVM__penalty': ['l1','l2'],
             'linSVM__loss': ['squared_hinge'],
              'linSVM__dual':[False]},
              {'linSVM__C': np.logspace(-3, 3, 8),
             'linSVM__penalty': ['l2'],
             'linSVM__loss': ['hinge'],
              'linSVM__dual':[True]}
             ],
    'NystroemSvm':{'nystroem__n_components': [50, 180, 400],
                   'nystroem__gamma': np.logspace(-3, 2, 6),
                   'svm__C': np.logspace(-4, 2, 10),
                   'svm__penalty': ['l2', 'l1'],
                   'svm__loss': ['squared_hinge']},
    'logistic':{'logistic__C': np.logspace(-4, 2, 8),
                'logistic__penalty':["elasticnet"],
                'logistic__l1_ratio':[0.0, 0.5, 1.0]
               }

}

res_dic = {}
scaler = ml_utils.SelectiveScaler(scalecols = ['num_ingr'])
for est in tqdm(estimators.keys()):
    p = Pipeline(steps=[('scaler', scaler)] + estimators[est])
    params = parameter_dic[est]
      
    X = X_tfidf_train.copy(deep=True)
    y = y_train.cuisine.copy(deep=True).to_numpy().ravel()
    print('Beginning to fit estimator {}'.format(est))
    starttime = timeit.default_timer()

    if est != 'logistic':
        res_dic[est] = GridSearchCV(p,params, n_jobs=-1,
                                    scoring='f1_micro',
                                    cv=5,
                                    pre_dispatch=2*3).fit(X, y)

    else:

        res_dic[est] = HalvingGridSearchCV(estimator=p,
                                            param_grid=params,
                                            factor=2,
                                            cv=5,
                                            scoring ='f1_micro',
                                            random_state=1234567,
                                            n_jobs=-1).fit(X,y)
    print('Fitted estimator {} in {} minutes'.format(est,
                                                     (timeit.default_timer()-starttime)/60
                                                    )
         )
    with open('../models/gridsearch_results_tfidf.p', 'wb') as handle:
        pickle.dump(res_dic, handle)

print('done...')
