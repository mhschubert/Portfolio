import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import numpy as np
import math

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn import svm, preprocessing, tree, ensemble


import seaborn as sns

from models import ModelType



#taken from https://james-brennan.github.io/posts/lowess_conf/
def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    if type(x) == type([]):
        x = np.array(x)

    if type(y) == type([]):
        y = np.array(y)


    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr


#taken and adapted from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def heatmap(data, row_labels, col_labels, normalize_axis = False, ax=None,
            cbar_kw={}, cbarlabel="", rotate=True, plot_cbar=True, **kwargs):
    """
    Create a heatmap from a np array and two lists of labels.

    Parameters
    ----------
    data
        A 2D np array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()


    if type(normalize_axis) == type(int(1)):
        data = normalize(data, axis=normalize_axis, norm='l1')
        #mn = data.min(axis=normalize_axis)[:, np.newaxis]
        #data = (data-mn)/(data.max(axis=normalize_axis)[:, np.newaxis] - mn)
    elif normalize_axis == 'total':
        mn = data.min()
        data = (data-mn)/(data.max() - mn)
    else:
        pass


    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if plot_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.026, pad=0.04,**cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30 * rotate, ha="right" if rotate else "center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def silouette_plot(X, cluster_labels, name, center_indices = False):
    
    if type(cluster_labels) != type(np.array([1])):
        cluster_labels = np.array(cluster_labels)
    n_clusters = len(np.unique(cluster_labels))

    # Create a subplot with 1 row and  2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)


    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])


    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        

        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    
    # 2nd Plot showing the actual clusters formed
    X = PCA(2).fit_transform(X)

    if type(center_indices) != type(False):
        centers = X[center_indices, :]
    else:
        #pick random centers
        lab_list = list(cluster_labels.flatten())
        center_indices = [lab_list.index(x) for x in np.unique(lab_list)]
        centers = X[center_indices, :]
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

    # Labeling the clusters

        

    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')


    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for {} clustering on sample data "
                  "with n_clusters = {}".format(name, n_clusters)),
                 fontsize=14, fontweight='bold')

    return fig


def plot_clustered_relevance(org, labels, relevance_array, y_predicted, target, Rs_dic,
    exclude, idx, mask_FP= False,  plot_trait_distrib = True):

    X = org.copy(deep=True)

    if type(mask_FP) == type(False):
        if not mask_FP:
            mask_FP = [True for _ in range(0, relevance_array.shape[0])]

    for label in np.unique(labels):
        if label != -1:
            label_mask = labels == label
            relevance_mean = normalize(np.mean(relevance_array[mask_FP & label_mask],
                                                  axis=0).reshape(-1,1), axis=0).reshape(-1,)
            if exclude:
                cols = [lab for i, lab in enumerate(list(X.columns)) if i in idx]
            else:
                cols = list(X.columns)
                
            pred_s = np.array(y_predicted)[mask_FP & label_mask]

            y_true_s = np.array(Rs_dic['group_viol']['FP']['ys'])[mask_FP & label_mask]

            #set fig dimensions 

            if plot_trait_distrib:
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,18))
            else:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,9))

            print(type(axs))
            axs[0,0].barh(cols, relevance_mean, height=1)
            axs[0,0].set_xlabel('Avg. Feature Importance')
            axs[0,0].set_ylabel('Features')
            axs[0,0].set_title('Cluster {}: Average Feature Importance'.format(label))
            
            y_correct = [y_pred == y_true for y_pred, y_true in zip(list(pred_s), list(y_true_s))]
            print("--Correct:", sum(y_correct))
            print("--Wrong:  ", len(y_true_s) - sum(y_correct))
            
            
            mat = confusion_matrix(y_true_s, pred_s)
            im, cbar = heatmap(mat, ["not_" + target, target], ["not_" + target, target], ax=axs[0,1],
                                           cmap="YlGn", cbarlabel="n_samples")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            axs[0,1].set_xlabel('Predicted', fontweight='bold')
            axs[0,1].set_ylabel('True', fontweight='bold')
            #axs.set_title('Prediction Results on ' + target + ' for ' + text, fontsize=20, fontweight='bold',
            #                  loc='left')
            #plt.show()
            if plot_trait_distrib:

                rel_dic = dict(zip([str(el) for el in list(relevance_mean)], cols))

                axs[1,:] = plot_trait_distribution(org, labels, axs=axs[1,:], cluster = label,
                    relevance_dic=rel_dic)
            print("########################################################################################")
            
            
            
    return fig


def plot_trait_distribution(org, labels, axs = None, cluster=None, relevance_dic = None):
    #define the variable names and what type of variable they encode
    boolish = ['is_misdem', 'p_current_on_probation', 'race_black', 'race_white', 'race_hispanic',
       'race_asian', 'race_native', 'is_married', 'is_divorced', 'is_widowed',
       'is_separated', 'is_sig_other', 'is_marit_unknown', 'is_male']

    count = ['offenses_within_30', 'p_felony_count_person', 'p_misdem_count_person',
       'p_charge_violent', 'p_juv_fel_count', 'p_felprop_violarrest', 'p_murder_arrest',
       'p_felassault_arrest', 'p_misdemassault_arrest', 'p_sex_arrest',
       'p_weapons_arrest', 'p_n_on_probation', 'p_prob_revoke', 'p_arrest', 'p_jail30', 'p_prison30', 'p_prison',
       'p_probation']

    numeric_scale = ['p_current_age', 'p_age_first_offense']

    scores = ['Risk of Recidivism_decile_score/10', 'Risk of Recidivism_raw_score',
       'Risk of Violence_decile_score/10', 'Risk of Violence_raw_score']   


    X = org.copy(deep=True) 
    X.loc[:, [el for el in scores if '/10' in el]] = X.loc[:, [el for el in scores if '/10' in el]]*10


    X['cluster'] = list(labels)
    if cluster or cluster == 0:
        if type(cluster) != type([]):
            cluster = [cluster]
        X = X.loc[X.cluster.isin(cluster), :]
 

    #plot only for one specific cluster
    if type(axs) == type(None):
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(24,18))

    X_agg = X.loc[:, X.columns.isin(boolish + ['cluster'])].groupby('cluster').apply(lambda g: (g.sum()/g.count())*100)
    #print(X_agg.head())

    tmp = X.loc[:, X.columns.isin(count + numeric_scale + scores + ['cluster'])].groupby('cluster').mean()
    
    X_agg = pd.concat([X_agg, tmp], axis = 1)
    X_agg.drop(['cluster'], inplace=True, axis=1)
    #print(X_agg.head())
   
    # if cluster:
    #     tmp = X_agg.loc[cluster, boolish].sort_index(axis=0).transpose().to_numpy()
    #     tmp2 = X_agg.loc[cluster, scores+count+numeric_scale].transpose().sort_index(axis=0).to_numpy()

    # else:
    #     tmp = X_agg.loc[:, boolish].sort_index(axis=0).transpose().to_numpy()
    #     tmp2 = X_agg.loc[:, scores+count+numeric_scale].transpose().sort_index(axis=0).to_numpy()

    # n1 = tmp.shape[0]
    # n2 = tmp2.shape[0]
    # if cluster:
    #     w = .25
    # else:
    #     w = 0.05

    # x = np.arange(0, tmp.shape[1])
    if cluster:

        tmp = X_agg.loc[X_agg.index == cluster, X_agg.columns.isin(boolish)].sort_index(axis=0).transpose().sort_index(axis=0)

        tmp2 = X_agg.loc[X_agg.index == cluster, ~X_agg.columns.isin(boolish)].transpose().sort_index(axis=0)

    else:
        tmp = X_agg.loc[:, X_agg.columns.isin(boolish)].sort_index(axis=0).transpose().sort_index(axis=0)
        tmp2 = X_agg.loc[:, ~X_agg.columns.isin(boolish)].transpose().sort_index(axis=0)
        
    tmpempty = tmp.empty
    tmp2empty = tmp2.empty
    if not tmpempty:
        axs[0] = tmp.plot(kind='barh', ax=axs[0])
        axs[0].set_yticklabels([col for col in list(X_agg.columns) if col in boolish])
        axs[0].set_xlabel('Percentage')
        axs[0].set_ylabel('Feature')
        axs[0].legend()
        
    if not tmp2empty:
        axs[1] = tmp2.plot(kind='barh', ax=axs[1])
        axs[1].set_yticklabels([col for col in list(X_agg.columns) if col in scores+count+numeric_scale])
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Feature')
        axs[1].legend()

    if relevance_dic:
        relevances = [float(el) for el in list(relevance_dic.keys())]
        if len(relevances) >=5:
            top_5 = [str(el) for el in sorted(relevances, key=lambda x: abs(x))[-5:]]
            cols = [relevance_dic[key] for key in top_5]
        else:
            cols = list(relevance_dic.values())
            

        if not tmpempty:
            plt1_labs = axs[0].get_yticklabels()
            plt1_ind = [i for i, lab in enumerate(plt1_labs) if lab.get_text() in cols]
            for i in plt1_ind:
                plt1_labs[i].set_color("red")
        if not tmp2empty:
            plt2_labs = axs[1].get_yticklabels()
            plt2_ind = [i for i, lab in enumerate(plt2_labs) if lab.get_text() in cols]
            for i in plt2_ind:
                plt2_labs[i].set_color("red")

    
    return axs


def plot_relevance_bars(values, columns, horizontal=True, ylab="", order=True, ax=None):
    """
    Nice barchart, values are the importance scores of the columns and must be in the same order.
    """
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("talk")

    ax_given = ax is not None

    if order:
        cols_ranked = [c for _, c in sorted(zip(values, columns))][::-1]
        values_ranked = sorted(values)[::-1]
    else:
        cols_ranked = columns
        values_ranked = values

    #sns.set_color_codes("pastel")
    if horizontal:
        if not ax_given:
            f, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(y=values_ranked, x=cols_ranked, color="b")
        ax.tick_params(axis='x', labelrotation=90)
        ax.set(xlabel="", ylabel=ylab)
    else:
        if not ax_given:
            f, ax = plt.subplots(figsize=(6, 15))
        sns.barplot(x=values_ranked, y=cols_ranked, color="b")
        ax.set(ylabel="", xlabel=ylab)
    sns.despine(left=True, bottom=True)

    if not ax_given:
        plt.show()


def plot_feature_scores(selection_result_dict, horizontal=True):
    """Plot the feature importances of a feature selection run."""
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("talk")

    all_cols = selection_result_dict["avg_rank"].keys()
    avg_value = {k:[] for k in all_cols}  # average removal error of each feature
    for run in range(len(selection_result_dict["selected_columns_all"])):
        old_cols = list(all_cols)
        for iter in range(0, len(selection_result_dict["selected_columns_all"][run])):
            feat = list(set(old_cols)-set(selection_result_dict["selected_columns_all"][run][iter]))
            # print(feat)
            assert len(feat) == 1
            feat = feat[0]
            if iter != 0:
                # There is a predecessor itertion
                avg_value[feat].append(selection_result_dict["scores_all"][run][iter-1] - selection_result_dict["scores_all"][run][iter])
            old_cols.remove(feat)

        # last remaining column:
        avg_value[selection_result_dict["selected_columns_all"][run][-1][0]].append(selection_result_dict["scores_all"][run][-1] - selection_result_dict["empty_score"])

    cols = []
    values = []
    for k,v in avg_value.items():
        values.append(-np.mean(v))
        cols.append(k)

    plot_relevance_bars(values, cols, horizontal=horizontal, ylab="Avg. reduction in score when removed")


def plot_avg_score(selection_result_dict):
    """Plot the avg score of a feature selection run."""
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("talk")

    avg_rank = selection_result_dict["avg_rank"]
    scores_all = selection_result_dict["scores_all"]

    print("")
    print("Avg. Removal Rank (higher is better)")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    sorted_cols = sorted(list(selection_result_dict["avg_rank"].keys()), key=(lambda c: avg_rank[c]))
    for col in sorted_cols:
        print("%.3f\t%s" % (avg_rank[col], col))

    scores_mean = np.mean(np.array(scores_all), axis=0)
    scores_std = np.std(np.array(scores_all), axis=0)

    x_axis = np.arange(len(scores_mean))[::-1] + 1  # reversed order bc features are successively removed
    plt.plot(x_axis, scores_mean)
    plt.fill_between(x_axis, scores_mean-scores_std, scores_mean+scores_std, alpha=0.25)
    plt.title("Avg. Loss at Eact Iteration of SBS")
    plt.xlabel("# Features")
    plt.ylabel("Loss")
    plt.show()


def confusion_matrices(keys, subkeys, y_val_predicted, y_val, threshold_predicted=0.5, threshold_true=3, restrict_dataset=False):
    """
    @param keys: list of e.g. 'group_viol'
    @param subkeys: list of e.g. 'FP'
    @param threshold_predicted: largest value that marks a negative predicion (model)
    @param threshold_true: largest value that marks a negative prediction (ground truth)
    @return:
    """
    for key in keys:
        for subkey in subkeys:
            target = key + '_' + subkey
            print(target)

            if restrict_dataset:
                data_subkey = key + ('_P' if 'P' in subkey else '_N')
                y_val_sub = y_val[y_val[data_subkey]]
                print(len(y_val_predicted[target]), len(y_val))

                if len(y_val_predicted[target]) == len(y_val):
                    y_val_predicted[target] = np.array(y_val_predicted[target])[y_val[data_subkey]]
            else:
                y_val_sub = y_val

            y_val_predicted_t = [1 if y > threshold_predicted else 0 for y in y_val_predicted[target]]
            y_val_predicted["correct"] = [int(y_val_sub[target][i] == y_val_predicted_t[i]) for i in range(len(y_val_predicted_t))]

            print("--Correct:", sum(y_val_predicted["correct"]))
            print("--Wrong:  ", len(y_val_sub[target]) - sum(y_val_predicted["correct"]))

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,9))
            mat = confusion_matrix(y_val_sub[target], y_val_predicted_t)
            im, cbar = heatmap(mat, ["not_" + target, target], ["not_" + target, target], ax=axs,
                                           cmap="YlGn", cbarlabel="n_samples")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            axs.set_xlabel('Predicted', fontweight='bold')
            axs.set_ylabel('True', fontweight='bold')
            #axs.set_title('Prediction Results on ' + target + ' for ' + text, fontsize=20, fontweight='bold',
            #                  loc='left')
            plt.show()

            print("Column ratios: %.2f, %.2f" % (mat[0,0]/mat[1,0], mat[0,1]/mat[1,1]))

            # Show confusion matrices (true VS compas) for the off-diagonal cells
            subcategories = [[0,1], [1,0]]  # off-diagonals
            fig, axs = plt.subplots(nrows=1, ncols=len(subcategories), figsize=(12,9))
            for s, sub in enumerate(subcategories):
                if "viol" in key:
                    gt = 'recid_violent'
                    compas = 'Risk of Violence_decile_score'
                else:
                    gt = 'recid'
                    compas = 'Risk of Recidivism_decile_score'

                gt_data = y_val_sub[gt]  # true value
                compas_data = y_val_sub[compas] > threshold_true # compas predicted
                # selector selects where our prediction disagrees with compas
                selector = [y_val_predicted_t[i] == sub[0] and y_val_sub[target][i] == sub[1] for i in range(len(y_val_sub))]
                mat = np.zeros([2,2])
                for i in range(len(selector)):
                    if selector[i]:
                        g = int(int(gt_data[i]) == int(compas_data[i]))
                        c = int(compas_data[i])
                        mat[g,c] += 1

                im, cbar = heatmap(mat, ["False","True"], ["Negative","Posistive"], ax=axs[s],
                                               cmap="YlGn")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                axs[s].set_xlabel('COMPAS', fontweight='bold')
                axs[s].set_ylabel('Ground Truth', fontweight='bold')
            plt.show()


def one_confusion_matrix(true, predicted, title=None, cell_labels=None, save=False, file_postfix=""):
    """
    Here, only one matrix will be created.
    """
    sns.set_style("ticks")
    sns.set_context("talk")

    if cell_labels is None:
        cell_labels = ["", ""]

    fig, axs = plt.subplots(nrows=1, ncols=1)
    mat = confusion_matrix(true, predicted)
    im, cbar = heatmap(mat, cell_labels, cell_labels, ax=axs, rotate=False, plot_cbar=False, cmap="Blues")#, cbarlabel="n_samples")
    annotate_heatmap(im, valfmt="{x:d}")
    axs.set_xlabel('Predicted')
    axs.set_ylabel('True')
    if title:
        axs.set_title(title)

    print("Column ratios: %.2f, %.2f" % (mat[0,0]/mat[1,0], mat[0,1]/mat[1,1]))

    plt.tight_layout()

    if save:
        plt.savefig("../../paper/COMPAS Reengineering/figures/confusion_%s.pdf" % file_postfix, dpi=300, bbox_inches='tight')

    plt.show()
    
    
def visualize_relevance(labels, relevance, target_label):
    plt.figure(figsize=(8,8))
    plt.barh(labels, relevance, height=1)
    plt.title(target_label)
    plt.xlabel("Linear model coefficients");
    plt.show()
    
def visualize_model(model_type: ModelType, model, X_train):
    if model_type == ModelType.NN:
        pass
    elif model_type == ModelType.SVM:
        visualize_relevance(X_train.columns, model.coef_[0], target)
        
    elif model_type == ModelType.DT:
        plt.figure(figsize=(14,14))
        tree.plot_tree(model, feature_names=X_train.columns, filled=True)
        plt.show()
        
    elif model_type == ModelType.RF:
        visualize_relevance(X_train.columns, model.feature_importances_, target)
