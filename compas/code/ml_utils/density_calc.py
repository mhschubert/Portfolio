import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency
import pandas as pd

def prob_occurence(oclist, cat=10):
    if type(oclist) != type([]) and type(oclist) == type(np.array([1,2])):
        oclist = list(oclist)
    c = Counter(oclist)
    keys = sorted(list(c.keys()))
    arr = np.zeros(cat)
    s = 0
    for i, key in enumerate(keys):
        s += c[key]
        arr[int(key)] = c[key]
    arr = arr/s
    return arr



def calc_multi_chi(frame, collist):
    ind = []
    for col in range(len(collist)):
        ind.append(frame[collist[col]].to_numpy())

    tab =  pd.crosstab(ind[0:(len(ind)-1)], ind[-1])
    #tab = frame.pivot_table(index=collist[0:(len(collist)-1)], columns=collist[-1], aggfunc={collist[-1]: len}, fill_value=0)
    #tab = frame.groupby(collist)[collist[-1]].count().unstack().fillna(0)

    names = tab.index.names
    #print(names)
    

    if len(collist) >2:
        
        ##make sure that all levels are present in index
        vals = tab.index.values

        lvls = len(vals[0])
        tmp_hold = []
        for i in range(lvls):
            tmp_hold.append([])
            for val in vals:
                tmp_hold[i].append(val[i])
            tmp_hold[i] = list(np.unique(tmp_hold[i]))
        mt = pd.MultiIndex.from_product(tmp_hold, names=names)

        tmp = []
        for ind in mt:
            if ind not in tab.index:
                tmp.append(ind)
        if len(tmp) > 0:
            for ind in tmp:
                tab.loc[ind, :] = 0
        tab = tab.sort_index()

        #calc shape of array
        tmp = []
        for _ in names:
            tmp.append([])
        for val_tup in tab.index.values:
            #print(val_tup)
            for i, val in enumerate(val_tup):
                tmp[i].append(val)
        for i in range(len(tmp)):
            tmp[i] = len(list(np.unique(tmp[i])))
        tmp = tuple(tmp+[len(tab.columns.values)])
        #print(tab)
        #make array and reshape
        #print(tab)
        arr = tab.to_numpy().reshape(tmp)
        arr = chi2_contingency(arr)
    else:
        arr = chi2_contingency(frame.groupby(collist)[collist[-1]].count().unstack().fillna(0).to_numpy())

    return arr




def contigency_table(oclist, cat=10):
    if type(oclist) != type([]) and type(oclist) == type(np.array([1,2])):
        oclist = list(oclist)  
    c = Counter(oclist)
    keys = sorted(list(c.keys()))
    arr = np.zeros(cat)
    for i, key in enumerate(keys):
        arr[int(key)] = c[key]
        
    return arr
    
    
    
