import os
import numpy
import pandas as pd
import ml_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def reorder_select_cols(orderdic, matchdic, datacols, tkey):
    # define input layer for each subscale
    # get input names from dic
    key = [key for key in list(orderdic.keys()) if matchdic[tkey].lower() in key.lower()]
    if len(key) != 1:
        raise ValueError('Too many keys to unpack')

    key = key[0]

    # ordered dic - always in the same order
    keys = []
    cols = []
    lens = []
    for k, v in orderdic[key].items():
        keys.append(k)
        cols.extend(v)
        lens.append(len(v))

    # get not usable columns for remainder set
    complement = [col for key in orderdic.keys() for col in orderdic[key].keys()]
    complement = [col for sublist in complement for col in sublist]

    print('Use {}'.format(keys))
    remainder = list(numpy.setdiff1d(datacols, complement))
    lens.append(len(remainder))
    cols.extend(remainder)

    return cols, lens


def load_and_prep_data(data_path, dataset="", clean_instead_drop=True):
    # Load data
    in_file = os.path.join(data_path, "input_features" + dataset + ".csv")
    out_file = os.path.join(data_path, "y_features" + dataset + ".csv")
    df_X = pd.read_csv(in_file, index_col="uid")
    df_y = pd.read_csv(out_file, index_col="uid")

    if clean_instead_drop:
        df_X = clean_data_of_na(df_X)
    else:
        df_X = drop_nas(df_X)

    matchframe = make_data_compatible(list(df_X.index.values), data_path, dataset)

    # Define targets
    df_y["Risk of Recidivism_decile_score"] = df_y["Risk of Recidivism_decile_score"] -1
    df_y["Risk of Violence_decile_score"] = df_y["Risk of Violence_decile_score"] - 1

    df_y["Risk of Recidivism_decile_score/10"] = df_y["Risk of Recidivism_decile_score"] / 10
    df_y["Risk of Violence_decile_score/10"] = df_y["Risk of Violence_decile_score"] / 10

    df_y["target_error_regression_base"] = abs(df_y["Risk of Recidivism_decile_score/10"] - df_y["recid"])
    df_y["target_error_regression_viol"] = abs(df_y["Risk of Violence_decile_score/10"] - df_y["recid"])

    # Remove index
    df_X.drop(columns='Unnamed: 0', inplace=True)
    # Remove 'p_famviol_arrest' since always 0
    # df_X.drop(columns='p_famviol_arrest', inplace=True)

    # Convert sex to boolean
    # print(df_X["sex"].unique())
    df_X["is_male"] = (df_X["sex"] == "Male")
    df_X.drop(columns="sex", inplace=True)

    df_X.drop(columns="first_offense_date", inplace=True)
    df_X.drop(columns="current_offense_date", inplace=True)

    return df_X, df_y, matchframe


def clean_data_of_na(df):
    df['offenses_within_30'].fillna(0, inplace=True)
    df['p_age_first_offense'].fillna(0, inplace=True)
    df['p_current_on_probation'].fillna(0, inplace=True)
    df['p_prob_revoke'].fillna(0, inplace=True)
    df['is_misdem'].fillna(0, inplace=True)

    return df


def drop_nas(df):
    df['offenses_within_30'].fillna(0, inplace=True)
    df['p_age_first_offense'].fillna(0, inplace=True)
    df.drop(['is_misdem'], inplace=True, axis=1)
    df.dropna(inplace=True)

    return df


def make_data_compatible(ind, data_path, dataset):
    if "reduced_size" not in dataset and "_pP" in dataset:
        out_file = os.path.join(data_path, "y_features_pP_recid_reduced_size.csv")
        df_r = pd.read_csv(out_file, index_col="uid")
        indices = list(df_r.index.values)
        del df_r

        bl = [True if i in indices else False for i in ind]

    elif "reduced_size" not in dataset and "_pP" not in dataset:
        out_file = os.path.join(data_path, "y_features_reduced_size.csv")
        df_r = pd.read_csv(out_file, index_col="uid")
        indices = list(df_r.index.values)
        del df_r

        bl = [True if i in indices else False for i in ind]

    else:
        bl = [True for i in ind]
    matchframe = pd.DataFrame({'uid': ind, 'is_reduced': bl})
    matchframe.set_index('uid', inplace=True)

    return matchframe

def make_cutoff(df, ran, col, true, fun):

    maxi = 0
    ct = 0
    for c in ran:
        pred = [1 if el > c else 0 for el in df[col]]
        sc = fun(df[true], pred)
        if sc >= maxi:
            maxi = sc
            ct = c
    print('The maximum score ({}) is {}'.format(fun.__name__, maxi))
    print('The cutoff selected is {}'.format(ct))
    return ct


def make_raw_to_decile(df_y, cutofftype, column):
    ##follows documentation that we want a mostly even distribution across deciles
    ##norm group is whole data set, i.e. we make an even distribution across data set

    score = {'acc': accuracy_score,
             'prec': precision_score,
             'rec': recall_score,
             'f1': f1_score}

    df_y['Risk of Recidivism_raw_to_decile_score'] = pd.qcut(df_y['Risk of Recidivism_raw_score'], q=10, precision= 3, labels=[0, 1,2,3,4,5,6,7,8,9])
    df_y['Risk of Violence_raw_to_decile_score'] = pd.qcut(df_y['Risk of Violence_raw_score'], q=10, precision=3,
                                          labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    #find cutoff via the cutofftype, i.e. acc, prec, rec, f1 (also a own function may be passed which returns the appropriate arguements

    ##select the column from which to calculate the optimal cutoff
    if 'Recidivism'.lower() in column.lower():
        col = 'Risk of Recidivism_raw_to_decile_score'
        true = 'recid'
    else:
        col = 'Risk of Violence_raw_to_decile_score'
        true = 'recid_violent'

    #select selector function
    if type('s') == type(cutofftype):
        fun = score[cutofftype]
    else:
        fun = cutofftype
    cutoff = make_cutoff(df_y, range(1,11), col, true, fun)




    return df_y, cutoff, 'raw_to_decile_score'


class Data_set:
    def __init__(self, base_path, dataset, clean_instead_drop = True, doubled_items= ['p_charge', 'p_famviol_arrest', "crim_inv_arrest",
                                                                               "crim_inv_charge", "vio_hist", "history_noncomp"],
                 overwrite = False):
        self.base_path = base_path
        self.data_path = os.path.join(base_path, "data", "preprocessed")
        self.dataset = dataset
        self.clean = clean_instead_drop
        self.doubled_items = doubled_items
        self.overwrite = overwrite

        self.df_X, self.df_y, self.matchframe = load_and_prep_data(self.data_path,
                                                                   dataset=self.dataset,
                                                                   clean_instead_drop=self.clean)
        self.df_X.drop(labels=self.doubled_items, axis=1, inplace=True)

    def setup_data(self, cutoff = 4, cuttype = 'decile_score',
                   min_val=0, max_val=1, strat=None, variable_dependent=False,
                   scorelabels=['Risk of Failure to Appear_decile_score',
                                'Risk of Failure to Appear_raw_score',
                                'Risk of Recidivism_decile_score', 'Risk of Recidivism_raw_score',
                                'Risk of Violence_decile_score', 'Risk of Violence_raw_score',
                                'Risk of Recidivism_decile_score/10',
                                'Risk of Violence_decile_score/10', 'target_error_regression_base',
                                'target_error_regression_viol']  # 'recid','recid_violent',
                   ):

        #check whether we make our own decile scores
        if 'raw' in cuttype:
            self.cutoff_criterium = cutoff
            self.raw_col = cuttype
            if type(self.cutoff_criterium) == int:
                cutoff = 'f1'
            self.df_y, cutoff, cuttype = make_raw_to_decile(self.df_y, cutoff, cuttype)
            if type(self.cutoff_criterium) == int:
                print('Reset to given cutoff {}'.format(self.cutoff_criterium))
                cutoff = self.cutoff_criterium
            self.own_decile = cuttype
            scorelabels = scorelabels + ['Risk of Recidivism_raw_to_decile_score',
                                         'Risk of Violence_raw_to_decile_score']

        # make rescaled and unrescaled data
        self.min_vals = {c: min(self.df_X[c]) for c in self.df_X.columns}  # for back-scaling
        self.max_vals = {c: max(self.df_X[c]) for c in self.df_X.columns}  # for back-scaling


        self.df_X_norm = ml_utils.rescale_df(self.df_X, min_val, max_val, variable_dependent)
        self.jail_info = self.df_X_norm.loc[:,['recid_corr', 'recid_violent_corr', 'custody_days_jail',
                                        'custody_days_jail_viol', 'jail_sentences',
                                        'jail_sentences_viol', 'custody_days_prison',
                                        'prison_sentences', 'custody_days_prison_viol', 'prison_sentences_viol']]

        self.df_X_norm.drop(['recid_corr', 'recid_violent_corr', 'custody_days_jail',
                                        'custody_days_jail_viol', 'jail_sentences',
                                        'jail_sentences_viol', 'custody_days_prison',
                                        'prison_sentences', 'custody_days_prison_viol', 'prison_sentences_viol'],
                            axis=1, inplace=True)
        if not os.path.exists(os.path.join(self.base_path, "data", "train_val_test", "{}_x_train.json".format(self.dataset))) or self.overwrite:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = ml_utils.custom_train_test_split(self.df_X_norm, self.df_y,
                                                                                              self.matchframe,
                                                                                              self.dataset,
                                                                                              test_size=0.25,
                                                                                              stratify=strat,
                                                                                              random_state=0)

            self.X_train.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_train.json".format(self.dataset)))
            self.X_val.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_val.json".format(self.dataset)))
            self.X_test.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_test.json".format(self.dataset)))
            self.y_train.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_train.json".format(self.dataset)))
            self.y_val.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_val.json".format(self.dataset)))
            self.y_test.to_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_test.json".format(self.dataset)))
        else:
            self.X_train = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_train.json".format(self.dataset)))
            self.X_val = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_val.json".format(self.dataset)))
            self.X_test = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_x_test.json".format(self.dataset)))
            self.y_train = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_train.json".format(self.dataset)))
            self.y_val = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_val.json".format(self.dataset)))
            self.y_test = pd.read_json(os.path.join(self.base_path, "data", "train_val_test", "{}_y_test.json".format(self.dataset)))


        self.featurelabels = list(self.df_X_norm.columns.values)

        self.ys_train = ml_utils.rescale_df(self.y_train[scorelabels], min_val, max_val, variable_dependent)
        self.y_train_norm = pd.concat([self.ys_train, self.y_train.drop(scorelabels + ['Unnamed: 0'], axis=1)], axis=1)

        self.ys_val = ml_utils.rescale_df(self.y_val[scorelabels], min_val, max_val, variable_dependent)
        self.y_val_norm = pd.concat([self.ys_val, self.y_val.drop(scorelabels + ['Unnamed: 0'], axis=1)], axis=1)

        self.ys_test = ml_utils.rescale_df(self.y_test[scorelabels], min_val, max_val, variable_dependent)
        self.y_test_norm = pd.concat([self.ys_test, self.y_test.drop(scorelabels + ['Unnamed: 0'], axis=1)], axis=1)

        self.set_groups(cutoff, cuttype)
        self.set_bool()
        self.make_index()

        print('done with setting up the data set...')

    def select_data(self,
                    y_to_X = ['Risk of Violence_decile_score/10', 'Risk of Violence_raw_score', ],
                    keys = ['group_viol'], subkeys= ['FP'] , subset =['p_misdem_count_person', 'p_current_age',
                                                                      'p_felassault_arrest', 'p_arrest',
                  'is_male'] ):
        X_train = self.X_train.copy(deep=True)
        X_val = self.X_val.copy(deep=True)
        X_test = self.X_test.copy(deep=True)
        y_train = self.y_train.copy(deep=True)
        y_val =self.y_val.copy(deep=True)
        y_test = self.y_test.copy(deep=True)
        for y2X in y_to_X:
            X_train.loc[:,y2X] = y_train[y2X]
            X_val.loc[:,y2X] = y_val[y2X]


        return X_train, X_val, X_test, y_train, y_val, y_test

    def marker_select(self, col, recid, cutoff, typ):
        if 'score' in typ:

            return ml_utils.iterative_marker(col, recid, cutoff)
        else:
            return ml_utils.marker(col, recid)


    def set_groups(self, cutoff=4, typ = 'decile_score'):

        col_viol = 'Risk of Violence_'+typ
        col_com = 'Risk of Recidivism_'+typ
        if 'score' in typ:
            print('use iterative marker...')
        else:
            print('use label-based marker...')
        self.y_train_norm.loc[:,'group_viol'] = self.y_train.apply(
            lambda x: self.marker_select(x[col_viol], x.recid_violent, cutoff, typ), axis=1)
        self.y_train.loc[:,'group_viol'] = self.y_train_norm['group_viol']

        self.y_val_norm.loc[:,'group_viol'] = self.y_val.apply(
            lambda x: self.marker_select(x[col_viol],x.recid_violent, cutoff, typ), axis=1)
        self.y_val.loc[:,'group_viol'] = self.y_val_norm['group_viol']

        self.y_test_norm.loc[:,'group_viol'] = self.y_test.apply(
            lambda x: self.marker_select(x[col_viol],x.recid_violent, cutoff, typ), axis=1)
        self.y_test.loc[:,'group_viol'] = self.y_test_norm['group_viol']


        self.y_train_norm.loc[:,'group_recid'] = self.y_train.apply(
            lambda x: self.marker_select(x[col_com], x.recid, cutoff, typ), axis=1)
        self.y_train.loc[:,'group_recid']= self.y_train_norm['group_recid']

        self.y_test_norm.loc[:,'group_recid'] = self.y_test.apply(
            lambda x: self.marker_select(x[col_com], x.recid, cutoff, typ), axis=1)
        self.y_test.loc[:,'group_recid'] = self.y_test_norm['group_recid']

        self.y_val_norm.loc[:,'group_recid'] = self.y_val.apply(
            lambda x: self.marker_select(x[col_com], x.recid, cutoff, typ), axis=1)

        self.y_val.loc[:,'group_recid'] = self.y_val_norm['group_recid']

    ##make subgroups
    def set_bool(self):
        for tp in ['group_viol', 'group_recid']:
            self.y_val.loc[:,tp + '_FP'] = self.y_val_norm[tp] == 1
            self.y_val.loc[:,tp + '_FN'] = self.y_val_norm[tp] == -1
            self.y_val.loc[:,tp + '_TP'] = self.y_val_norm[tp] == 0.5
            self.y_val.loc[:,tp + '_TN'] = self.y_val_norm[tp] == -0.5
            self.y_val.loc[:,tp + '_T'] = abs(self.y_val_norm[tp]) < 1
            self.y_val.loc[:,tp + '_P'] = self.y_val_norm[tp] > 0
            self.y_val.loc[:,tp + '_N'] = self.y_val_norm[tp] < 0
            
            self.y_train.loc[:,tp + '_FP'] = self.y_train_norm[tp] == 1
            self.y_train.loc[:,tp + '_FN'] = self.y_train_norm[tp] == -1
            self.y_train.loc[:,tp + '_TP'] = self.y_train_norm[tp] == 0.5
            self.y_train.loc[:,tp + '_TN'] = self.y_train_norm[tp] == -0.5
            self.y_train.loc[:,tp + '_T'] = abs(self.y_train_norm[tp]) < 1
            self.y_train.loc[:,tp + '_P'] = self.y_train_norm[tp] > 0
            self.y_train.loc[:,tp + '_N'] = self.y_train_norm[tp] < 0
            
            self.y_test.loc[:,tp + '_FP'] = self.y_test_norm[tp] == 1
            self.y_test.loc[:,tp + '_FN'] = self.y_test_norm[tp] == -1
            self.y_test.loc[:,tp + '_TP'] = self.y_test_norm[tp] == 0.5
            self.y_test.loc[:,tp + '_TN'] = self.y_test_norm[tp] == -0.5
            self.y_test.loc[:,tp + '_T'] = abs(self.y_test_norm[tp]) < 1
            self.y_test.loc[:,tp + '_P'] = self.y_test_norm[tp] > 0
            self.y_test.loc[:,tp + '_N'] = self.y_test_norm[tp] < 0

    def make_index(self):
        self.index_val = {}
        self.index_train = {}
        self.index_test = {}
        for tp in ['group_viol', 'group_recid']:
            self.index_val[tp] = {}
            self.index_train[tp] = {}
            self.index_test[tp] = {}
            for direct, val in [('false_negative', -1), ('true_negative', -.5),
                                ('true_positive', .5), ('false_positive', 1)]:
                self.index_val[tp][direct] = list(self.y_val_norm.loc[self.y_val_norm[tp] == val].index)

                self.index_train[tp][direct] = list(self.y_train_norm.loc[self.y_train_norm[tp] == val].index)
                self.index_test[tp][direct] = list(self.y_test_norm.loc[self.y_test_norm[tp] == val].index)


        #flatten

        self.index_val_flat = {}
        self.index_train_flat = {}
        self.index_test_flat = {}
        for key in self.index_val.keys():
            self.index_val_flat[key] = {'negative': [], 'positive': []}
            self.index_train_flat[key] = {'negative': [], 'positive': []}
            self.index_test_flat[key] = {'negative': [], 'positive': []}
            for subkey in self.index_val[key].keys():
                if 'negative' in subkey:
                    self.index_val_flat[key]['negative'].extend(self.index_val[key][subkey])
                    self.index_train_flat[key]['negative'].extend(self.index_train[key][subkey])
                    self.index_test_flat[key]['negative'].extend(self.index_test[key][subkey])
                else:
                    self.index_val_flat[key]['positive'].extend(self.index_val[key][subkey])
                    self.index_train_flat[key]['positive'].extend(self.index_train[key][subkey])
                    self.index_test_flat[key]['positive'].extend(self.index_test[key][subkey])






