
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders

def run_pca():
    # constants
    input_filepath = '../../data/datasets/2_19/risk_10_sec_10_timesteps.h5'
    debug_size = 100000
    timesteps = 1
    target_index = None
    n_components = 2
    batch_size = 1000

    # load data
    data = dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=False, train_split=1., 
        debug_size=debug_size, timesteps=timesteps, target_index=target_index)
    features = data['x_train']
    targets = data['y_train']
    idxs = np.where(np.sum(targets, axis=1) > 0)[0]
    features = features[idxs]
    targets = targets[idxs]
   
    # run pca
    # pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    # plot it
    reduced_features = pca.transform(features)
    colors = []
    counts = collections.defaultdict(int)
    cat_max = 500
    idxs = []
    for i, t in enumerate(targets):
        c = None
        if t[0] > 0. and counts[0] < cat_max:
            c = 'blue'
            counts[0] += 1
        elif t[1] > 0. and counts[1] < cat_max:
            c = 'red'
            counts[1] += 1
        elif t[2] > 0. and counts[2] < cat_max:
            c = 'purple'
            counts[2] += 1
        elif t[3] > 0. and counts[3] < cat_max:
            c = 'orange'
            counts[3] += 1
        elif t[4] > 0. and counts[4] < cat_max:
            c = 'green'
            counts[4] += 1
        if c is not None:
            colors.append(c)
            idxs.append(i)
    plt.figure(figsize=(10,10))
    plt.scatter(reduced_features[idxs,0], reduced_features[idxs,1], c=colors, alpha=.5)
    plt.show()

if __name__ == '__main__':
    run_pca()