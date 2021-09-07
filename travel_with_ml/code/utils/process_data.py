import numpy as np
import pandas as pd
import helper
import copy
class Data(object):
    def __init__(self, path:str):
        self.path = path
        self.df = pd.read_csv(path, header=0)

    def clean_data(self, datacols:list):
        # drop rows with all zeros in the data columns
        self.df_clean = copy.deepcopy(self.df.loc[(self.df[datacols] != 0).all(axis=1)])
        #we have some terretories for which we have the home countries capital (i.e. double appearance)
        self.df_clean.drop_duplicates(subset=['CapitalName'], inplace=True)
        self.df_clean.drop_duplicates(subset=['CapitalLatitude', 'CapitalLongitude'], inplace = True)

    def make_dist_matrix(self, longcol:str, latcol:str, converter:helper.Haversine):
        rank = self.df_clean.shape[0]
        matrix = np.zeros(shape=(rank,rank))
        low_tri_ind = np.tril_indices_from(matrix, k=-1)
        for comb in zip(*low_tri_ind):
            cit1 = self.df_clean.iloc[[comb[0]]]
            cit2 = self.df_clean.iloc[[comb[1]]]
            coord1 = (float(cit1[longcol]), float(cit1[latcol]))
            coord2 = (float(cit2[longcol]), float(cit2[latcol]))
            matrix[comb[0], comb[1]] = converter.calc_distance((coord1), (coord2))
        # make symmetric
        matrix.T[low_tri_ind] = matrix[low_tri_ind]
        np.fill_diagonal(matrix, float('inf'))
        self.matrix = matrix
