
import argparse
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import os
from sklearn import dummy
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import multioutput
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
import sys
import time

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders

MODEL_TYPES = [
    'linear_regression', 
    'random_forests', 
    'gradient_boosting',
    'extra_trees',
    'bagging',
    'adaboost',
    'neural_network',
    # 'svm',
    'constant_mean',
    'constant_median', 
    'constant_zero']

def fit(model, data):
    print("fitting {}".format(model))
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    model.fit(x_train, y_train)
    r_sq = model.score(x_val, y_val)

    idxs = get_nonzero_idxs(y_val)[:5]
    y_pred = model.predict(x_val[idxs])
    y = y_val[idxs]
    print(y_pred)
    print(y)
    print(r_sq)
    return r_sq

def build_model(model_type, num_targets = 1):
    if model_type == 'linear_regression':
        base = linear_model.SGDRegressor()
    elif model_type == 'random_forests':
        base = ensemble.RandomForestRegressor()
    elif model_type == 'gradient_boosting':
        base = ensemble.GradientBoostingRegressor()
    elif model_type == 'extra_trees':
        base = ensemble.ExtraTreesRegressor()
    elif model_type == 'bagging':
        base = ensemble.BaggingRegressor()
    elif model_type == 'adaboost':
        base = ensemble.AdaBoostRegressor()
    elif model_type == 'neural_network':
        base = neural_network.MLPRegressor()
    elif model_type == 'svm':
        base = svm.SVR(verbose=1)
    elif model_type == 'constant_mean':
        base = dummy.DummyRegressor('mean')
    elif model_type == 'constant_median':
        base = dummy.DummyRegressor('median')
    elif model_type == 'constant_zero':
        base = dummy.DummyRegressor('constant', constant=0)
    else:
        raise(ValueError('invalid model type: {}'.format(model_type)))

    # multiple outputs in the dataset => fit a separate regressor to each
    if num_targets > 1:
        return multioutput.MultiOutputRegressor(base)
    else:
        return base

def get_nonzero_idxs(y):
    if len(y.shape) > 1:
        _, num_targets = y.shape
        idxs = np.hstack(np.where(y[:,i] != 0)[0] for i in range(num_targets))
    else:
        idxs = np.where(y != 0)[0]
    
    return list(idxs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_type', 
        default='all')
    parser.add_argument('-f', dest='dataset_filepath', 
        default='../../data/datasets/1_1/risk_26.h5')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # in case things get a bit crazy
    np.random.seed(1)

    # parse inputs
    opts = parse_args()

    # load the dataset
    data = dataset_loaders.risk_dataset_loader(opts.dataset_filepath, 
        normalize=True, debug_size=None, train_split=.9, shuffle=True)

    # build the model
    if len(data['y_train'].shape) > 1:
        _, num_targets = data['y_train'].shape
    else:
        num_targets = 1
    if opts.model_type == 'all':
        models = [build_model(mt, num_targets) for mt in MODEL_TYPES]
    else:
        model = build_model(opts.model_type, num_targets)



    # x, y = data['x_train'], data['y_train']
    # idxs_train = get_nonzero_idxs(y)
    # x, y = data['x_val'], data['y_val']
    # idxs_val = get_nonzero_idxs(y)

    # print(idxs_train)
    # print(idxs_val)
    # input()

    # data['x_train'] = data['x_train'][idxs_train]
    # data['y_train'] = data['y_train'][idxs_train]
    # data['x_val'] = data['x_val'][idxs_val]
    # data['y_val'] = data['y_val'][idxs_val]
    
    # fit the model
    st = time.time()
    if opts.model_type == 'all':
        results = [fit(m, data) for m in models]
    else:
        results = fit(model, data)
    et = time.time()
    print('model fitting took {} seconds'.format(et - st))

    # display results
    if opts.model_type == 'all':
        results = sorted(zip(results, MODEL_TYPES), reverse=True)
        for (r, l) in results:
            print('{}: {}'.format(l,r))
    else:
        print(results)
