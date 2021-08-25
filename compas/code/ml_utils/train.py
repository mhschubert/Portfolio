from collections import OrderedDict
import torch
import numpy as np
import sklearn.metrics
from sklearn import svm, tree, ensemble
from matplotlib import pyplot as plt
import sys
import ml_utils
from datasets import PandasDataset
from models import Custom_Net, ModelType


def get_target_type(target):
    is_list = False
    if type(target) in (list, tuple):
        # For now, we assume that multi-target components are regression targets
        is_binary, is_multiclass, is_regression = [0, 0, 1]
        is_list = 1
    elif target in ["Risk of Recidivism_raw_score", "target_error_regression", "Risk of Violence_raw_score",
                  "Risk of Recidivism_decile_score/10", "Risk of Violence_decile_score/10",
                  "target_error_regression_viol"]:
        is_binary, is_multiclass, is_regression = [0, 0, 1]
    elif target in ["Risk of Recidivism_decile_score", "Risk of Violence_decile_score", "residual_binned"]:
        is_binary, is_multiclass, is_regression = [0, 1, 0]
    elif target in ['recid', 'recid_proPub', 'recid_violent', 'recid_violent_proPub', "target_error_binary_base",
                    "target_error_binary_viol"] or target.endswith("_FP") or target.endswith("_FN") or target.endswith("_T") or target.endswith("FH") or target.endswith("FL") or target.endswith("binary"):
        is_binary, is_multiclass, is_regression = [1, 0, 0]
    else:
        raise ValueError('Could not establish target type.')

    return is_binary, is_multiclass, is_regression, is_list


def fit_NN(X_train, X_test, y_train, y_test, target, scales, regex_keys, n_layers, n_hidden, n_epochs, weighting=None, weighting_table=None,
           batch_size=16, bias=False, activation=torch.nn.ReLU, weight_decay=0.0, lr=1e-3, device="cpu", netparas=None,
           return_loss=False, return_validation_loss=False, validate=True):
    print("Using device", device)

    # Define target type and cardinality of the NN output
    is_binary, is_multiclass, is_regression, is_list = get_target_type(target)
    if is_list:
        n_out = len(target)
    else:
        n_out = 1 if is_binary or is_regression else len(y_train[target].unique())
    n_in = len(X_train.columns)

    # Define NN structure
    if type(scales) == type(OrderedDict([('test', 'test')])):
        if is_list:
            raise NotImplementedError("Scales have not been implemented for multiple targets.") #TODO
        # order columns in dataset
        ordered_cols, lens = ml_utils.reorder_select_cols(scales, regex_keys, X_train.columns, target)
        print(X_train.columns)

        X_train = X_train[ordered_cols]
        X_test = X_test[ordered_cols]

        if netparas is None:
            netparas = {'bias': True,
                        'scalewidth': 'singular',
                        'enc': False
                        }
        model = Custom_Net(inputwidth_list=lens, n_layers=n_layers, n_hidden=n_hidden,
                           enc=netparas['enc'],
                           outwidth=n_out, bias=netparas['bias'],
                           scalewidth=netparas['scalewidth']).to(device)

    else:
        layers = []
        for l in range(n_layers):
            layers += [torch.nn.Linear(n_in if l == 0 else n_hidden, n_hidden, bias=bias), activation()]
        layers += [torch.nn.Linear(n_hidden, n_out)]

        model = torch.nn.Sequential(*layers).to(device)

    #calc weight table if needed
    if type(weighting) == dict:
        weights = ml_utils.make_weight_table(y_train[weighting[target]])
        print("Weights:", weights)
    elif type(weighting) == list:
        X_train, y_train = ml_utils.oversample(X_train=X_train, y_train=y_train, target=target, weighting=weighting)
        weights = None
        weighting = None
        print("Warning: Weighting is disabled!")
        # weighting = {target: target}
    elif type(weighting_table) == dict:
        weights = weighting_table
    else:
        weights = None
        weighting = None
        # weighting = {target: target}
    
    # Define data loader & optimizer
    if weighting is not None:
        tr_loader = torch.utils.data.DataLoader(PandasDataset(X_train, y_train[target],
                                                          y_train[weighting[target]]), batch_size=batch_size,
                                            shuffle=True)
    else:
        tr_loader = torch.utils.data.DataLoader(PandasDataset(X_train, y_train[target],
                                                          y_train[target]), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if is_regression:
        weight = 1
    else:
        if weights is None:
            weight = None
        else:
            weight = torch.tensor([weights[float(li)] for li in range(len(weights))], requires_grad=False)
            print(weight)

    if is_multiclass:
        loss_name = "CE"
        loss_function = lambda y_predicted, y_true, weight_per_class: torch.nn.CrossEntropyLoss(weight=None)(y_predicted, y_true[:,0].long())
    elif is_binary:
        loss_name = "BCEwL"
        loss_function = lambda y_predicted, y_true, weight_per_class: torch.nn.BCEWithLogitsLoss(weight=(weight_per_class[1]/weight_per_class[0]) if weight_per_class is not None else None)(y_predicted, y_true)
    else:
        loss_name = "RMSE"
        loss_function = lambda y_predicted, y_true, weight: torch.sqrt(torch.nn.MSELoss()(y_predicted, y_true))*weight

    losses = []
    # Fitting loop
    for epoch in range(n_epochs):
        print("Epoch: %d/%d" % (epoch + 1, n_epochs))
        cumul_loss = []
        for (x, y, l) in tr_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)[:, 0]
            
            #check for loss weighting
            if weights:
                if is_regression:
                    weight = [weights[float(li)] for li in list(l)]
                    weight = float(np.sum(weight)/len(weight))

            if is_list:
                loss_val = sum([loss_function(y_hat[:,y_i], y[:,0,y_i], weight) for y_i in range(len(target))])
            else:
                loss_val = loss_function(y_hat, y, weight)
                #print(y_hat, y, weight, loss_val)
                #import pdb
                #pdb.set_trace()
            loss_val.backward()
            optimizer.step()
            cumul_loss.append(loss_val.detach())#/weight)
        avg_loss = float(torch.mean(torch.stack(cumul_loss)).cpu().numpy())
        print("Train %s %.3f:" % (loss_name, avg_loss))
        losses.append(avg_loss)

    # Evaluate
    if validate:
        if is_list:
            y_test_predicted = []
            for t_i, t in enumerate(target):
                model_sub = lambda x: model(x)[..., t_i]
                y_test_predicted.append(evaluate(model_sub, X_test, y_test, t, device))
        else:
            y_test_predicted, val_loss = evaluate(model, X_test, y_test, target, device, return_loss=True)
    else:
        y_test_predicted = None

    to_return = {"model" : model,
                 "y_hat" :y_test_predicted}
    if return_loss:
        to_return["loss_train"] = losses
    if return_validation_loss and validate:
        if is_list:
            raise NotImplementedError("return_validation_loss not implemented for list-type targets.")
        to_return["loss_val"] = val_loss
    return to_return


def evaluate(model, X_test, y_test, target, device="cpu", return_loss=False):
    is_binary, is_multiclass, is_regression, is_list = get_target_type(target)

    val_loader = torch.utils.data.DataLoader(PandasDataset(X_test, y_test[target], y_test[target]), batch_size=1,
                                             shuffle=False)

    if is_regression:
        loss_functions_names = ["RMSE"]
        loss_functions = [lambda y_predicted, y_true: torch.sqrt(torch.nn.MSELoss()(y_predicted, y_true))]
    elif is_multiclass:
        loss_functions_names = ["RMSE", "CE"]
        loss_functions = [lambda y_predicted, y_true: torch.sqrt(torch.nn.MSELoss()(y_predicted, torch.argmax(y_true, 1))),
                          lambda y_predicted, y_true: torch.nn.CrossEntropyLoss()(y_predicted, y_true[:,0].long())]
    elif is_binary:
        # TODO implement
        loss_functions_names = ["RMSE", "BCEwL"]
        loss_functions = [lambda y_predicted, y_true: torch.sqrt(torch.nn.MSELoss()(y_predicted, torch.argmax(y_true, 1))),
                          lambda y_predicted, y_true: torch.nn.BCEWithLogitsLoss()(y_predicted, y_true)]
        #raise NotImplementedError("No binary loss-function implemented yet.")
    else:
        raise ValueError("Target variable of unknown cardinality/type.")

    cumul_loss = []
    y_test_predicted = []
    y_all = []
    y_hat_all = []
    for (x, y, l) in val_loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)[:, 0]
        y_test_predicted.append(y_hat.cpu().detach())
        if is_binary:
            # loss_val = torch.sum(y_hat.gt(0.5)==y, dtype=torch.float)
            y_all.append(y.cpu().detach())
            y_hat_all.append(y_hat.cpu().detach())
            loss_vals = [loss_function(y_hat, y).detach().cpu() for loss_function in loss_functions]
            cumul_loss.append(loss_vals)
        else:
            loss_vals = [loss_function(y_hat, y).detach().cpu() for loss_function in loss_functions]
            cumul_loss.append(loss_vals)
    if is_binary:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_all, y_hat_all)
        auc = sklearn.metrics.auc(fpr, tpr)
        print("AUC:", auc)
        avg_losses = np.mean(np.array(cumul_loss), axis=0)
        for l_i, loss in enumerate(avg_losses):
            print("Validation %s %.3f:" % (loss_functions_names[l_i], loss))
    else:
        avg_losses = np.mean(np.array(cumul_loss), axis=0)
        for l_i, loss in enumerate(avg_losses):
            print("Validation %s %.3f:" % (loss_functions_names[l_i], loss))

    if return_loss:
        return y_test_predicted, avg_losses
    else:
        return


def fit_any(model_type: ModelType, X_train, X_val, y_train, y_val, target, config, weighting=None, device="cpu", plot_loss=False):
    if model_type == ModelType.NN:
        if weighting:
            w = {target: target}
        else:
            w = None

        results = fit_NN(X_train, X_val, y_train, y_val, target, regex_keys=None, scales=False,
                                         device=device,
                                         weighting_table=w,
                                         validate=False,
                                         return_loss=plot_loss,
                                         **config)
        model = results["model"]

        val_loader = torch.utils.data.DataLoader(PandasDataset(X_val, y_val, y_val), batch_size=1, shuffle=False)
        y_val_predicted = [model(xy[0])[:, 0].cpu().detach() for xy in val_loader]

        if plot_loss:
            plt.plot(results["loss_train"])
            plt.show()

    else:
        if weighting is not None:
            # Define weighting
            n_ones = sum(y_train[target])
            n_zeros = len(y_train[target]) - n_ones
            weight0 = (len(y_train)/2) / n_zeros
            weight1 = (len(y_train)/2) / n_ones

        # Train model
        if model_type == ModelType.SVM:
            if weighting:
                w = [(weight1 if y == 1 else weight0) for y in y_train[target]]
            else:
                w = None

            model = svm.LinearSVC(**config)
            model.fit(X_train, y_train[target].astype(int), sample_weight=w)

        elif model_type == ModelType.DT:
            if weighting:
                w = {0:weight0, 1:weight1*weighting}
            else:
                w = None

            model = tree.DecisionTreeClassifier(class_weight=w, **config)
            model.fit(X_train, y_train[target].astype(int))

        elif model_type == ModelType.RF:
            if weighting:
                w = {0:weight0, 1:weight1*weighting}
            else:
                w = None
            model = ensemble.RandomForestClassifier(class_weight=w, **config)
            model.fit(X_train, y_train[target].astype(int))

        # Evaluate
        y_val_predicted = model.predict(X_val).astype(float)

    # calculate error terms
    y_val_predicted_bin = np.array([1 if y > (0 if model_type == ModelType.NN else 0.5) else 0 for y in y_val_predicted])
    correct = np.array([int(y_val[target][i] == y_val_predicted_bin[i]) for i in range(len(y_val_predicted_bin))])
    acc = sum(correct) / len(correct)

    return model, y_val_predicted, acc




def misclassification_difference_binary(pred, y_true, expected, weighting=(1, 1, 1)):
    # we need to predict all categories at once so that can optimize left and right side of cutoff at the same time

    # tensor has shape (samples, class) [0 or 1] Positive or Negative
    # y_true has the shape (samples, [COMPAS, TRUE]); the predictions are on ground truth
    pred = pred.reshape((-1,1))
    pred = torch.sigmoid(pred)
    y_true = y_true.reshape((pred.size()[0], 2))
    ground_truth = y_true[:,1].reshape((-1,1))
    gr_inverse = 1.0 - ground_truth
    errors = torch.tensor([torch.abs(y_true[i,0]-y_true[i,1]) for i in range(0,y_true.shape[0])]).reshape((-1,1))
    #errors = torch.tensor(y_true[:,0] != y_true[:,1]).reshape((-1,1))##find false positives and false negatives
    diff = errors - pred


    mse_loss = torch.nn.MSELoss()(pred, errors)
    correction = torch.ge(pred, 0.5).float().reshape((-1,1))
    correction = y_true[:,0].reshape((-1,1)) - correction
    #acc = torch.mean(torch.eq(correction-y_true[:,1].reshape((-1,1,)),0).float())
    #print(acc)
    errors = errors.bool()
    FN= torch.mean(torch.abs(diff)*(ground_truth*errors.float() + ground_truth*(~errors).float())) # (FN + TP) * abs(errors - predictions) #FN
    FP = torch.mean(torch.abs(diff)* (gr_inverse*errors.float() + gr_inverse*(~errors).float())) # (FP + TN) * abs(errors - predictions) #FP




    #sys.exit()

    loss = weighting[0]*((expected[1] - FP)**2)+ weighting[1]*((expected[3] - FN)**2)+(mse_loss)*weighting[2]

    #loss = mse_loss

    return loss



def fit(X_train, X_val, y_train, y_val, expected,
        n_layers, n_hidden, n_epochs, n_out, loss_function= lambda y_predicted, y_true, exp, weight: misclassification_difference_binary(y_predicted,
                                                                                                   y_true,
                                                                                                   exp,
                                                                                                   weight),
        weight=(1,1, 1),
        batch_size=500, bias=False, activation=torch.nn.ReLU, weight_decay=0.0, lr=1e-3, device="cpu",
        return_loss=False, return_validation_loss=False, validate=True, vocal = 0):


        print("Using device", device)

        # Define target type and cardinality of the NN output
        n_out = n_out
        n_in = len(X_train.columns)

        # Define NN structure

        layers = []
        for l in range(n_layers):
            layers += [torch.nn.Linear(n_in if l == 0 else n_hidden, n_hidden, bias=bias), activation()]
        layers += [torch.nn.Linear(n_hidden, n_out)]

        model = torch.nn.Sequential(*layers).to(device)

        # Define data loader & optimizer
        tr_loader = torch.utils.data.DataLoader(PandasDataset(X_train, y_train), batch_size=batch_size,
                                                    shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        loss_function = loss_function

        losses = []
        # Fitting loop
        for epoch in range(n_epochs):
            if vocal == 1:
                print("Epoch: %d/%d" % (epoch + 1, n_epochs))
            cumul_loss = []
            for (x, y) in tr_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_hat = model(x)[:, 0]

                #sys.exit()

                # check for loss weighting

                loss_val = loss_function(y_hat, y, expected, weight)
                # print(y_hat, y, weight, loss_val)
                # import pdb
                # pdb.set_trace()
                loss_val.backward()
                optimizer.step()
                cumul_loss.append(loss_val.detach())  # /weight)
            avg_loss = float(torch.mean(torch.stack(cumul_loss)).cpu().numpy())
            if vocal == 1:
                print("Train %.3f:" % (avg_loss))
            losses.append(avg_loss)
        print("Final Train-Loss %.3f:" % (avg_loss))

        # Evaluate
        if validate:
            y_test_predicted = evaluate(model, X_val, y_val, expected, weight, loss_function,
                                                  device,
                                                  return_loss=True)

            val_loss = None
        else:
            y_test_predicted = None

        to_return = {"model": model,
                     "y_hat": y_test_predicted}
        if return_loss:
            to_return["loss_train"] = losses
        if return_validation_loss and validate:

            to_return["loss_val"] = val_loss
        return to_return




def evaluate(model, X_test, y_test, expected, weight, loss_function,
             device="cpu", return_loss=False):

    val_loader = torch.utils.data.DataLoader(PandasDataset(X_test, y_test), batch_size=1,
                                             shuffle=False)

    loss_function = loss_function

    cumul_loss = []
    y_test_predicted = []
    y_all = []
    y_hat_all = []
    for (x, y) in val_loader:

        if not 0 in np.array(y.size()):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)[:,:]
            y_test_predicted.append(y_hat.cpu().detach())
            #loss_vals = loss_function(y_hat, y, expected, weight)
            #cumul_loss.append(loss_vals.detach().numpy())

    #avg_losses = np.mean(np.array(cumul_loss))

    #print("Validation %.3f:" % (avg_losses))

    if return_loss:
        return y_test_predicted#, avg_losses
    else:
        return

def eval(model, X_test, y_test, device="cpu"):

    val_loader = torch.utils.data.DataLoader(PandasDataset(X_test, y_test), batch_size=1,
                                             shuffle=False)


    y_test_predicted = []
    for (x, y) in val_loader:

        if not 0 in np.array(y.size()):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)[:,:]
            y_test_predicted.append(y_hat.cpu().detach())
            #loss_vals = loss_function(y_hat, y, expected, weight)
            #cumul_loss.append(loss_vals.detach().numpy())

    #avg_losses = np.mean(np.array(cumul_loss))

    #print("Validation %.3f:" % (avg_losses))


    return y_test_predicted
