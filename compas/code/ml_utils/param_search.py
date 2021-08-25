import copy
import random
from collections import OrderedDict
import torch
import numpy as np

import train
from models import ModelType

_default_spaces = {
    ModelType.NN: OrderedDict([
        ("activation", [torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid]),
        ("n_layers", [i for i in range(1, 5)]),
        ("n_hidden", [i * 10 for i in range(1, 6)]),
        ("batch_size", [2 ** i for i in range(8, 14, 2)]),
        ("lr", [0.01, 0.001]),
        ("weight_decay", [0.0, 0.001]),
        ("n_epochs", [100])
    ]),
    ModelType.DT: OrderedDict([
        ("criterion", ["gini", "entropy"]),
        ("max_depth", [2, 3, 4])
    ]),
    ModelType.RF: OrderedDict([
        ("n_estimators", [i * 100 for i in range(1, 5)]),
        ("criterion", ["gini", "entropy"]),
        ("max_depth", [2, 3, 4])
    ]),
    ModelType.SVM: OrderedDict([
        ("penalty", ["l1", "l2"]),
        ("C", [0.001, 0.1, 1., 10, 50]),
        ("dual", [False])
    ]),
}


def get_space(model_type: ModelType):
    return copy.deepcopy(_default_spaces[model_type])


def calc_n_combinations(space):
    """Returns the number of cnfigurations contained in a search space."""
    n = 1
    for k,v in space.items():
        n *= len(v)
    return n


def get_config(space, id):
    """Returns the id'th configuration from the search space."""
    config = {}
    space_skipped = 0
    space_size = calc_n_combinations(space)
    for k,v in space.items():
        for v_i in range(len(v)):
            if id < space_skipped + (v_i+1)*space_size/len(v):
                config[k] = v[v_i]
                space_size = space_size/len(v)
                space_skipped += (v_i)*space_size
                break
    return config


def search_space(model_type: ModelType, X_train, X_val, y_train, y_val, target, n_models, weighting=None, space=None,
                 device="cpu", seed=0):
    """Searches the given parameter space by randomly sampling n_models configurations and evaluating their accuracy."""
    if space is None:
        space = get_space(model_type)

    n_possible = calc_n_combinations(space)
    print("#possible configurations", n_possible)
    if n_models > n_possible:
        n_models = n_possible
        print("Reduced #model to train to", n_models)

    # Sample configurations:
    random.seed(seed)
    configs = []
    idcs = []
    for c in range(n_models):
        # Pick a random config and avoid duplicates
        if n_models < n_possible:
            idx = random.randint(0, n_possible-1)
            while idx in idcs:
                idx = random.randint(0, n_possible-1)
        else:
            idx = c
        idcs.append(idx)
        config = get_config(space, idx)
        configs.append(config)

    # Train & evaluate for every configuration
    scores = []
    for c, config in enumerate(configs):
        print("Training config %d/%d" % (c + 1, n_models))
        torch.manual_seed(seed)
        _, _, acc = train.fit_any(model_type, X_train, X_val, y_train, y_val, target, config, weighting, device)
        scores.append(acc)

    # Sort configs by score and print
    configs = [c for _, _, c in sorted(zip(scores, np.arange(n_models), configs), reverse=True)]
    scores = sorted(scores, reverse=True)
    for c, s in zip(configs, scores):
        print(s)
        print("%.3f\t" % s, [str(k) + str(v) for k, v in c.items()])

    return configs, scores
