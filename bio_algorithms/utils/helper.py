import numpy as np
import math

def split_path_unix_win(path):
    #make windows-unix problem go away
    if  '\\' in path:
        path = path.split('\\')
    elif '/' in path:
        path = path.split('/')

    return path

def get_nonzero_indices(arr:np.ndarray, condition=None):

    if type(condition) == type(None):
        inds = arr.nonzero()
    else:
        inds = (arr < condition).nonzero()
    return inds

def get_all_nonzeros(lst:list):
    inds = []
    for path in lst:
        inds.extend([el for el in zip(*get_nonzero_indices(arr=path))])
    uni = [[],[]]
    #get unique indices
    hash_map = set()
    for ind in inds:
        if str(ind) not in hash_map:
            uni[0].append(ind[0])
            uni[1].append(ind[1])
            hash_map.add(str(ind))
    return uni

def get_span_tree_simil(path:list, min_st:list):
    #always smaller city-index first for unique path (we do not care about direction here)
    min_path = set(['{}{}'.format(i,j) if i<j else '{}{}'.format(j,i) for i,j in zip(*min_st)])
    c = 0
    for i in range(1, len(path)):
        trans = '{}{}'.format(path[i-1], path[i]) if path[i-1]<path[i] else '{}{}'.format(path[i], path[i-1])
        if trans in min_path:
            c+=1
    return c/len(min_path)

class Haversine(object):
    def __init__(self, scale ='meter'):
        '''
        accepts followinf values as scale
        meter => meter
        km => kilometer
        mile => miles
        feet =>x feet
        '''
        self.scale = scale

    def calc_distance(self, coord1:tuple, coord2:tuple):
        long1, lat1 = coord1
        long2, lat2 = coord2

        earth = 6371000 #earth radius in meter
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)

        delta_phi = phi2 - phi1
        delta_lamb = math.radians(long2) - math.radians(long1)
        inner = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lamb/2)**2
        rad = 2 * math.atan2(inner**(1/2), (1-inner)**(1/2))

        dist = earth * rad

        return self._convert(dist)


    def _convert(self, dist:float):
        if self.scale == 'meter':
            return dist
        elif self.scale == 'kilometer':
            return dist/1000
        elif self.scale == 'mile':
            return dist * 0.000621371
        else:
            return dist * 3.28084