import copy
import os
import sys

import numpy as np

import train
from plot_helpers import plot_avg_score


def _fit_wrapper(*args):
    """Just returns the last return argument (scores) of the fit function"""
    return train.fit_any(*args)[-1]


def select_feat(base_path, X_train, X_val, y_train, y_val, target, model_type, config, weighting, device="cpu", n_runs=1, save=True, pool=None):
    """Itertelive removes one feature after the other."""
    # Pool needs to be passed in from outside, because it must be created where __name__=="__main__"

    scores_all = []
    selected_columns_all = []
    removed_cols_all = []

    for run in range(n_runs):
        print("############################ Run %d/%d ############################" % (run+1, n_runs))
        sys.stdout.flush()
        columns = X_train.columns
        n_iterations = len(columns)-1
        scores = []
        selected_columns = []
        removed_cols = []

        # Calculate reference score: full
        full_score = train.fit_any(model_type, X_train, X_val, y_train, y_val, target, config, weighting, device)[-1]

        for iter in range(n_iterations):
            print("Feature selection iteration %d/%d" % (iter + 1, n_iterations))
            scores_iter = [0]*(len(columns))
            reduced_cols_iter = []
            for c in range(len(columns)):
                reduced_cols = list(columns)
                reduced_cols = reduced_cols[:c] + reduced_cols[c+1:]
                reduced_cols_iter.append(reduced_cols)
                X_train_reduced = X_train[reduced_cols]
                X_val_reduced = X_val[reduced_cols]
                if pool is None:
                    scores_iter[c] = train.fit_any(model_type, X_train_reduced, X_val_reduced, y_train, y_val, target, config, weighting, device)[-1]
                else:
                    xtr = copy.deepcopy(X_train_reduced)
                    xvr = copy.deepcopy(X_val_reduced)
                    yt = copy.deepcopy(y_train)
                    yv = copy.deepcopy(y_val)
                    cf = copy.deepcopy(config)

                    scores_iter[c] = pool.apply_async(_fit_wrapper, (xtr, xvr, yt, yv,
                                                         target,
                                                         cf,
                                                         weighting,
                                                         device))  # return val acc

            best_iter_score = None
            best_iter = 0
            best_reduced_cols = None
            for c in range(len(columns)):
                if pool is not None:
                    scores_iter[c] = scores_iter[c].get()
                    print("finished %d" % (c + 1))
                    sys.stdout.flush()
                try:
                    # for multiple error values, choose the last
                    scores_iter[c] = scores_iter[c][-1]
                except:
                    pass

                print(columns[c], scores_iter[c])

                if best_iter_score is None or scores_iter[c] > best_iter_score:
                    best_iter_score = scores_iter[c]
                    best_iter = c
                    best_reduced_cols = reduced_cols_iter[c]

            scores.append(best_iter_score)
            selected_columns.append(best_reduced_cols)
            removed_cols.append(columns[best_iter])
            columns = best_reduced_cols
            if len(best_reduced_cols) == 1:
                break

        scores_all.append(scores)
        selected_columns_all.append(selected_columns)
        removed_cols_all.append(removed_cols)

    if pool is not None:
        pool.close()
        pool.join()

    # Calculate ranking of features, avg. over runs
    avg_rank = {}
    for rem_cols in removed_cols_all:  # iterates through runs
        for col_i, c in enumerate(rem_cols):  # iterates through removals
            if c in avg_rank:
                avg_rank[c] += (1+col_i)/n_runs  # avg removal-rank: higher is better
            else:
                avg_rank[c] = (1+col_i)/n_runs

    # Account for the case when a feature was never removed
    for c in X_train.columns:
        if c not in avg_rank:
            avg_rank[c] = len(X_train.columns)

    # Calculate reference scores: empty
    empty_score = None
    X_train["empty"] = 0.5
    X_val["empty"] = 0.5
    xt_empty = X_train[["empty"]]
    xv_empty = X_val[["empty"]]
    empty_score = train.fit_any(model_type, xt_empty, xv_empty, y_train, y_val, target, config, weighting, device)[-1]

    scores_mean = np.mean(np.array(scores_all), axis=0)

    to_return = {"scores_all": scores_all,
                       "scores_mean": scores_mean,
                       "avg_rank": avg_rank,
                       "selected_columns_all": selected_columns_all,
                        "full_score": full_score,
                 "empty_score": empty_score}

    if save:
        save_features(to_return, target, base_path)

    plot_avg_score(to_return)

    return to_return


def save_features(dic, target, base_path):
    print("Saving %s features." % target)
    path = os.path.join(base_path, "saves")
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, target + ".feat"), dic)


def load_features(target, base_path):
    print("Loading %s features." % target)
    path = os.path.join(base_path, "saves")
    feat = np.load(os.path.join(path, target + ".feat.npy"), allow_pickle=True)[()]
    return feat