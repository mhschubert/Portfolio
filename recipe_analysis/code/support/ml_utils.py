import nltk
import re
import spacy
import string
from nltk.stem import PorterStemmer
import numpy as np
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA,PCA,TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori
from IPython.display import display
from sklearn.svm import LinearSVC, SVC
##for models:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

import sankey_helper


import pandas as pd


class BrandRemover():
    ##uses the spacy NER to remove Brand names and other stuff from the ingredients
    ##expects the text ingredients to be in long format
    def __init__(self, col, drop_len=2):
        self.nlp = spacy.load("en_core_web_sm")
        self.tags = ["CARDINAL", "DATE", "GPE", "LOC", "MONEY", "ORDINAL", "ORG",
                     "PERCENT", "PRODUCT", "QUANTITY", "TIME"]

        self.drop_len = drop_len
        self.col =col

    def fit(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        # iterate over all ingredients
        X.reset_index(drop = True, inplace=True)
        to_del = []
        for i, ingredient in enumerate(X.loc[:, self.col]):
            # parse them into a doc
            doc = self.nlp(ingredient)
            # iterate over all ents
            # if ent empty use ingredient string as is
            if doc.ents == ():
                newstring = ingredient
            # else delete entity if it is a part of the tags specified above
            else:
                newstring = ingredient
                for ent in doc.ents:
                    newstring = newstring.replace(ent.text, "")
                # remove excessive blanks
                " ".join(newstring.split())
            X.loc[i, self.col] = newstring
            if len(newstring)< self.drop_len:
                to_del.append(i)
        if to_del:
            to_keep = set(range(X.shape[0])) - set(to_del)
            X = X.take(list(to_keep))
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X, y)


class Preprocessor():
    # expects the text ingredients to be in long format

    def __init__(self, idx, idy, col, min_recipe=3):
        self.idy = idy
        self.idx = idx
        self.min_recipe = min_recipe
        self.col = col
        self.german_ingr = {'zitrone': 'lemon', 'zitronen saft': 'lemon juice',
                            'zwiebel': 'onions', 'öl': 'oil'}

        self.stopwords = ['low[\s]*fat', 'reduced[\s]*fat', 'fat[\s]*free', 'non[\s]*fat', 'gluten[\s]*free', 'free[\s]*range',
                          'reduced[\s]*sodium', 'salt[\s]*free', 'sodium[\s]*free', 'low[\s]*sodium', 'sweetened', 'unsweetened',
                          'large', "all[\s]*purpose"
                          'extra[\s]*large', 'oz', '®', '™', 'oldelpaso', 'alfredo', 'knorr', 'all-purpose',
                          'hellmann', 'orbestfoodcanolacholesterolfree', 'heinz', 'kraft', 'lipton', 'taco bell',
                          '€', 'best foods', "uncle ben's", 'wishbone®', 'herdez', 'sargento® artisan blends®',
                          "oscar mayer cotto", "colman's", "i can't believe it's not butter!®", "everglades",
                          "jack daniels", "half\s*&half", "all[\s-]*purpos[e]*", "fresh", "large"
                          "campbell's", "hellmann", "oz", "m&m", "pasoâ„¢", "soy vay®"]

        self.spelling = [['tumeric', 'turmeric'], ['yoghurt', 'yogurt'], ['yogurt', 'yogurt'],
                         ['fillet', 'raw fish'], ['mozzarella', 'mozzarella cheese'],
                         ['chile', 'chili'], ['chili', 'chili'], ['chilies', 'chili'],
                         ['chilli', 'chili'], ['sriracha', 'chili'], ['romain', 'romaine',],
                         ['lettuc', 'lettuce'], ['oliv', 'olive']]

        self.remove_chopchop = re.compile(r'crushed|crumbles|ground|minced|powder|chopped|sliced')
        self.numeric = re.compile('[0-9]')
        self.min_len = 2
        self.symbols = ['®', '€', '$']
        #find all characters not giving us info
        self.remove_digits = str.maketrans('', '', string.digits)
        self.remove_punctuation = str.maketrans('', '', string.punctuation)
        self.remove_whitespace = str.maketrans('', '', string.whitespace)

        self.substitution = re.compile(r'[^\w\s]')  ##finds words starting at beginning of string
        self.stemmer = PorterStemmer()

    #for compatibility with scikit pipeline
    def transform(self, X, y=None):
        return self.preprocess(X, y)

    def fit(self, X, y=None):
        return self.transform(X, y)

    def fit_transformt(self, X, y=None):
        return self.transform(X, y)

    def cumulate_ingredients(self, df, aggcol, col):
        # save a list of ingredients to per row per ceipe
        # use numpy here to save lots of time

        keys, values = df.loc[:, [aggcol, col]].sort_values(aggcol).values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        df = pd.DataFrame({aggcol: ukeys, col: [list(a) for a in arrays]})
        return df

    def preprocess(self, X, y):

        to_del = []
        X.reset_index(drop = True, inplace=True)
        for i, ingr in enumerate(X.loc[:, self.col]):
            txt = ingr.lower()
            txt = txt.replace('-', '')
            #replace german words
            for key in self.german_ingr.keys():
                txt = re.sub(key, self.german_ingr[key], txt)

            #fix some spelling errors
            for cword in self.spelling:
                txt = re.sub(cword[0], cword[1], txt)
            #replace the random nonsenical stuff
            for spword in self.stopwords:
                txt = re.sub(spword, "", txt)

            #remove chopchop
            txt = self.remove_chopchop.sub('', txt)

            # remove all digits and parenthesis content from ingredients
            txt = txt.translate(self.remove_punctuation).translate(self.remove_digits)
            txt = re.sub(r'\(*\)', '', txt)
            # check whether string contains nonsensical characters when removing all of those with regular expression

            if re.findall('[^a-zA-Z\s]', self.substitution.sub('', txt)):
                to_del.append(i)

            ##check min_len
            if len(txt)<=self.min_len:
                to_del.append(i)

            # do the stemming
            fin = []
            for word in txt.split():
                if len(word) > 0:
                    fin.append(self.stemmer.stem(re.sub(r'[^\w\s]', '', word)))
            fin = ' '.join(fin)

            X.loc[i, self.col] = fin

        #lets drop every ingredient which is not really an ingredient after all
        #we use set operations here because so much fastness :)
        if to_del:
            to_keep = set(range(X.shape[0])) - set(to_del)
            X = X.take(list(to_keep))

        #save all ingredients per recipe to list
        X = self.cumulate_ingredients(df=X, aggcol = self.idx, col=self.col)
        X['num_ingr'] = list(X.apply(lambda row: len(row['ingredients']), axis=1))
        X = X.loc[X.num_ingr > self.min_recipe, :]
        y = y.loc[y[self.idy].isin(X[self.idx]),:]


        return X, y

def cumulate_ingredients(df, aggcol, col):
    # save a list of ingredients to per row per ceipe
    # use numpy here to save lots of time

    keys, values = df.loc[:, [aggcol, col]].sort_values(aggcol).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df = pd.DataFrame({aggcol: ukeys, col: [list(a) for a in arrays]})
    return df


class Featurizer():

    def __init__(self, idx, vectorize, feat_not_vectorize, min_df = None, reduced = True, max_features = None,
                 tfidf = False):

        self.not_vectorize = [n for n in feat_not_vectorize if n != idx]
        self.idx =idx
        self.feat_x = vectorize
        self.reduced = reduced
        if tfidf:
            self.vectorizer = TfidfVectorizer(preprocessor=self.identity_preprocessor,
                                          tokenizer=self.identity_tokenizer, min_df=min_df,
                                          max_features = max_features)

        else:
            self.vectorizer = CountVectorizer(preprocessor=self.identity_preprocessor,
                                              tokenizer=self.identity_tokenizer, min_df=min_df,
                                              max_features = max_features,
                                              binary=True)


    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.feat_x])



    def transform(self, X, y=None):
        dtm = self.vectorizer.transform(X[self.feat_x])
        display(dtm)
        keys = X.loc[:,self.idx_x]
        dtm = pd.DataFrame(dtm.todense(), columns=self.vectorizer.get_feature_names())
        dtm.index= list(keys)
        for k in self.not_vectorize:
            dtm[k] = list(X[k])

        return dtm

    def fit_transform(self,X, y=None):
        dtm = self.vectorizer.fit_transform(X[self.feat_x])
        keys = X.loc[:,self.idx]
        dtm = pd.DataFrame(dtm.todense(), columns=self.vectorizer.get_feature_names())
        dtm.index = list(keys)
        for k in self.not_vectorize:
            dtm[k] = list(X[k])

        return dtm

    def identity_tokenizer(self, x):
        return x

    def identity_preprocessor(self, x):
        return x


def get_tsne(df, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200, init="pca"):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter,
                learning_rate=learning_rate, init=init, random_state=1234567, n_jobs=-1)

    df = tsne.fit_transform(df)
    print(tsne.kl_divergence_)
    return df, tsne, tsne.kl_divergence_





def get_kmeans_wcss(data, n_limit=20, title=""):
    wcss = []  # Within cluster sum of squares (WCSS)
    for i in range(1, n_limit):
        km = KMeans(init='k-means++', n_clusters=i, n_init=10, n_jobs=1, random_state=1234567)
        km.fit(data)
        wcss.append(km.inertia_)
    plt.title("Elbow Method {}".format(title))
    plt.plot(range(1, n_limit), wcss)
    plt.xticks(range(1,n_limit))
    plt.grid(True)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
    return wcss


def kmeans(data, n):
    km = KMeans(init='k-means++', n_clusters=n, n_init=10, n_jobs=1, random_state=1234567)
    km = km.fit(data)
    return km.predict(data), km


def get_TSVD(df, n_components=2, n_iter=5, algorithm='randomized'):
    tsvd = TruncatedSVD(n_components=n_components, n_iter=n_iter, algorithm=algorithm)
    reduced_data = tsvd.fit_transform(df)
    explained_variance = tsvd.explained_variance_ratio_
    print(explained_variance)
    return reduced_data, tsvd, explained_variance


def create_tsne_graph(cluster_tsne, red_tsne, n_clus):
    c_mask = []
    c_x = []
    c_y = []

    for i in range(0, n_clus):
        c_mask.append([x for x in cluster_tsne == i])

    for i in range(0, n_clus):
        c_x.append([a[0] for a, b in zip(red_tsne, c_mask[i]) if b])
        c_y.append([a[1] for a, b in zip(red_tsne, c_mask[i]) if b])

    colours = ['magenta', 'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'black']

    for i in range(0, n_clus):
        plt.scatter(c_x[i], c_y[i], s=30, c=colours[i], label='Cluster {}'.format(i))

    plt.title("Clusters of tSNE")
    plt.xlabel("Feat 1")
    plt.ylabel("Feat 2")
    plt.legend()
    plt.show()

def get_regions():
    region_dic = {'brazilian':'american',
               'jamaican':'american',
               'mexican':'american',
               'southern_us':'american',
               'cajun_creole':'american',
               'spanish':'europe_south',
               'moroccan':'europe_south',
               'french':'europe_south',
               'british':'europe_north',
               'irish':'europe_north',
               'italian':'europe_south',
               'greek':'europe_south',
               'russian':'europe_north',
               'indian':'asian',
               'thai':'asian',
               'vietnamese':'asian',
               'chinese':'asian',
               'japanese':'asian',
               'korean':'asian',
               'filipino':'asian'

    }

    regions = {region:[] for cuisine, region in region_dic.items()}
    for cuisine, region in region_dic.items():
        regions[region].append(cuisine)

    return region_dic, regions

def plot_scatter_cuisines(X, labs, ax=None, sublabs=None,
                          palette={'brazilian':'rosybrown',
               'jamaican':'salmon',
               'mexican':'black',
               'southern_us':'darkred',
               'cajun_creole':'indianred',
               'spanish':'orangered',
               'moroccan':'yellow',
               'french':'tan',
               'british':'darkviolet',
               'irish':'fuchsia',
               'italian':'orange',
               'greek':'gold',
               'russian':'darkkhaki',
               'indian':'lime',
               'thai':'mediumspringgreen',
               'vietnamese':'mediumturquoise',
               'chinese':'darkcyan',
               'japanese':'aqua',
               'korean':'cadetblue',
               'filipino':'dodgerblue'
                                   }
                          ):

    gave_axis = True
    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=(10,12))
        gave_axis = False

    if type(sublabs) == type(None):
        labstodo = np.unique(labs)
    else:
        labstodo = sublabs

    for i, lab in enumerate(labstodo):
        mask = labs == lab
        ax.scatter(X[mask, 0], X[mask,1], label=str(lab), facecolors='none', edgecolors=palette[lab])

    if not gave_axis:
        print(palette)
        plt.legend()
        plt.show()
    else:
        return ax

def pairwise_frequencies(X):
    A = np.zeros((X.shape[1], X.shape[1])) ##pairwise frequency matrix A
    for ing_id in range(X.shape[1]):
        freq_vec = X[X[:, ing_id] == 1].sum(axis=0)
        freq_vec[ing_id] = 0
        A[ing_id] = list(freq_vec)

    return A



def get_freq_net_cuisine(X_one_hot, mask, tot_ing):
    #Computes the pairwise frequency between ingredients in a given cuisine
    tmp = X_one_hot[mask]
    ingr = list(range(0,tot_ing))

    cuisine_ing_net = np.zeros((tot_ing, tot_ing))


    # Iterate on the list of ingredients and compute their frequency
    for ing in ingr:

        freq_vec = tmp[tmp[:,ing]==1].sum(axis=0)

        # Frequency when it is paired with itself is removed (diagonal is zero)
        freq_vec[ing] = 0
        cuisine_ing_net[ing] = freq_vec
    return cuisine_ing_net



def build_ingredient_graph(X, y, X_one_hot, vectorizer,
                    palette = {'brazilian':'rosybrown',
                   'jamaican':'salmon',
                   'mexican':'black',
                   'southern_us':'darkred',
                   'cajun_creole':'indianred',
                   'spanish':'orangered',
                   'moroccan':'yellow',
                   'french':'tan',
                   'british':'darkviolet',
                   'irish':'fuchsia',
                   'italian':'orange',
                   'greek':'gold',
                   'russian':'darkkhaki',
                   'indian':'lime',
                   'thai':'mediumspringgreen',
                   'vietnamese':'mediumturquoise',
                   'chinese':'darkcyan',
                   'japanese':'aqua',
                   'korean':'cadetblue',
                   'filipino':'dodgerblue'

        },
                           label_col = ['ID_recipe', 'cuisine'],
                           figsize=(40,40)):
    '''This code was taken and adapted from https://github.com/alialamiidrissi/ADA_Course_Project/blob/master/Project/graph_helper.py'''

    cuisines = palette.keys()
    # Build the graph where each node is an ingredient
    # And edges link ingredients frequently used togheter in recipes
    G = nx.Graph()
    j = 0

    invert_vocab = {val:key for key, val in vectorizer.vocabulary_.items()}
    ##map dishes to cuisines
    mapping = X.merge(y, left_on='ID_recipe', right_on ='ID', how='left').loc[:, label_col]

    ing_net = pairwise_frequencies(X_one_hot)
    for cuisine in cuisines:

        mask = mapping[label_col[1]] == cuisine
        # Get the pairwise frequency matrix and discount it
        cuisine_ing_net = get_freq_net_cuisine(X_one_hot, mask, tot_ing= X_one_hot.shape[1])
        cuisine_net_discounted = cuisine_ing_net / (ing_net + 1)
        i = 0  # Add associated edges and nodes to the graph
        for raveled_id in np.argsort(-cuisine_net_discounted.reshape(-1)):
            a, b = np.unravel_index(raveled_id, cuisine_net_discounted.shape)
            # Take only a small number of ingredient for visual convinience
            if i > 10:
                break
            u, v = invert_vocab[a], invert_vocab[b]
            G.add_edge(u, v, cuisine=j, color=palette[cuisine], weight=1, width=cuisine_net_discounted[a, b])
            i += 1

        j += 1


    # Build a dictionary to associate each ingredient
    # to a cuisine where it appears the most
    partition = {}
    for u in G.nodes():
        best_color = {}
        for v in G[u]:
            best_color[G[u][v]['color']] = 1 if not G[u][v]['color'] in best_color else best_color[G[u][v]['color']] + 1
        partition[u] = sorted(best_color.items(), key=lambda x: x[1])[-1][0]

    # Compute the x,y coordinates of the nodes
    pos = community_layout(G, partition)

    # Plot the graph (nodes and edges)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=3)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Plot the legend
    patches = [mpatches.Patch(color=co, label=str(cu).title()) for co, cu in zip([palette[c] for c in cuisines], cuisines)]
    plt.legend(handles=patches)
    plt.title('Network of ingredients')
    plt.axis('off')



def community_layout(g, partition):
    #https: // stackoverflow.com / questions / 43541376 / how - to - draw - communities -
    #with-networkx / 43541777matplotlib % 20show % 20grid
    """
    Compute the layout for a modular graph.
    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot
    partition -- dict mapping int node -> int community
        graph partitions
    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions
    """

    pos_communities = _position_communities(g, partition, scale=4.5)

    pos_nodes = _position_nodes(g, partition, scale=1.3)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ax = None,
                          fig=None):
    """
    Discalimer: This snippet was taken from the official scikit-learn documentation
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if type(ax) == type(None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,14))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)


    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, cax=cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.show()



class basket_analyzer():
    def __init__(self):
        self.executed = False


    def error_message(self):
        print('You have to execute the algorithm on the data first')

    def get_measures(self, measures=['antecedents', 'consequents', 'support', 'confidence', 'lift'],
                     ascending =False,
                     sort_by='confidence',
                     get_top = 100):

        if not self.executed:
            self.error_message()

        result = self.rules[measures]

        if type(get_top) != type(int):
            get_top = len(result)
        return result.sort_values(sort_by, ascending=ascending)[0:get_top]


    def apriori(self, X_one_hot, min_support = 0.01, metric='lift', min_threshold=1.0):
        self.frequent_items = apriori(X_one_hot, min_support=min_support, use_colnames=True)
        self.rules = association_rules(self.frequent_items, metric=metric, min_threshold=min_threshold)
        self.executed = True



class Cuisine_Feat_Creator():

    def __init__(self, selectcol = 'cuisine', proportional = True, not_part=None):
        self.selectcol = selectcol
        self.proportional = proportional
        self.not_part = not_part

    def fit(self, X, y=None):
        tmp= X
        if type(self.not_part) != type(None):
            tmp = X.iloc[:, 0:(self.not_part-1)]

        self.cusine_vectors = tmp.groupby(by=y[self.selectcol]).apply(sum, axis=0)
        #encode one-hot
        self.cusine_vectors.loc[:,:] = np.where(self.cusine_vectors >0, 1,0)

    def transform(self, X,y=None):
        tmp = X
        if type(self.not_part) != type(None):
            tmp = X.iloc[:, 0:(self.not_part - 1)]

        res = pd.DataFrame(tmp.apply(axis=1, func = lambda row: np.where(np.array(self.cusine_vectors)+np.array(row).reshape((1,-1)) >1,
                                                    1,0).T.sum(axis=0)).to_list()).to_numpy()



        if self.proportional:
            res = np.array(res)/self.cusine_vectors.sum(axis=1).to_numpy().reshape((-1, self.cusine_vectors.shape[0]))

        for i, cus in enumerate(self.cusine_vectors.index.values):
            X[cus] = list(res[:,i])

        return X

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)


class SelectiveScaler(MinMaxScaler):
    def __init__(self, scalecols, feature_range=(0, 1), *, copy=True, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
        self.scalecols = scalecols



    def fit(self, X,y=None):
        return super().fit(X.loc[:, self.scalecols], y)

    def fit_transform(self, X, y=None, **fit_params):
        scaled =  super().fit_transform(X.loc[:, self.scalecols], y, **fit_params)
        X.loc[:,self.scalecols] = scaled
        return X.to_numpy()

    def transform(self, X):
        scaled = super().transform(X.loc[:, self.scalecols])
        X.loc[:,self.scalecols] = scaled
        return X.to_numpy()



class Evaler():

    def __init__(self, X, y, fun, params={}):
        self.X = X
        self.y = y
        self.fun = fun()
        for param in params.keys():
            setattr(self.fun, param, params[param])


    def evalfn(self, **kwargs):

        for key in kwargs:
            setattr(self.fun, key, float(kwargs[key]))
        fun = self.fun
        f = cross_val_score(fun, self.X, self.y, cv=5, scoring='f1_micro')
        return f.max()


class Eval():

    def __init__(self, X, y, fun, params={}):
        self.X = X
        self.y = y
        self.fun = fun()
        for param in params.keys():
            setattr(self.fun, param, params[param])


    def evalfn(self, **kwargs):

        for key in kwargs:
            setattr(self.fun, key, float(kwargs[key]))
        fun = self.fun
        f = cross_val_score(fun, self.X, self.y, cv=5, scoring='f1_micro')
        return f.max()




def plot_flow_chart(y_org, pred, cm, labels, normed=False):

    df_cm_norm = pd.DataFrame(data=cm, index=labels, columns=labels)
    if not normed:
        df_cm_norm = df_cm_norm / df_cm_norm.sum(axis=1)


    mask = np.not_equal(np.array(y_org), np.array(pred))

    y_org_labeled = list(y_org[mask])
    pred_labeled = list(np.array(pred)[mask])

    sankey_helper.sankey_plotter(y_org_labeled, pred_labeled, aspect=20, fontsize=11, cnf_matrix=df_cm_norm)
    plt.gcf().set_size_inches(10, 10)

