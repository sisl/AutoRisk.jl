
import collections
import h5py
import numpy as np

def normalize_features(data, threshold=1e-8):
    """
    Description:
        - Normalize the dataset (features).

    Args:
        - data: dictionary containing x_train and x_val
        - threshold: threshold for std dev at which 
            no division takes place

    Returns:
        - normalized dataset
    """
    if len(data['x_train'].shape) == 2:
        axes = 0
    else:
        axes = (0,1)

    mean = np.mean(data['x_train'], axis=axes)
    data['x_train'] = (data['x_train'] - mean)

    # compute the standard deviation after mean subtraction
    std = np.std(data['x_train'], axis=axes)

    # if the standard deviation is sufficiently low
    # then just divide by 1
    std[std < threshold] = 1

    # normalize
    data['x_train'] = data['x_train'] / std
    data['x_val'] = (data['x_val'] - mean) / std

    # store means and standard deviations as well
    data['means'] = mean
    data['stds'] = std

    return data

def discretize_targets(targets, num_bins):
    # linear bins for now
    cuts = np.linspace(0, 1, num_bins + 1)

    # collect indices where the target is between the cut
    bin_idxs = []
    for lo, hi in zip(cuts, cuts[1:]):
        idxs = np.where((targets >= lo) & (targets <= hi))
        bin_idxs.append(idxs)

    # for each bin, assign values that belong to it
    for c, idxs in enumerate(bin_idxs):
        targets[idxs] = c

def get_balanced_class_weights(targets):
    weights = np.empty(targets.shape)
    for tidx in range(targets.shape[1]):
        # count classes and normalize
        c = collections.Counter()
        c.update(targets[:,tidx])
        for k in c.keys():
            c[k] = c[k] ** -1
        max_v = max(c.values())
        for k in c.keys():
            c[k] /= max_v

        # insert weights
        for k in c.keys():
            idxs = np.where(targets[:,tidx] == k)[0]
            weights[idxs, tidx] = c[k]
    return weights

def risk_dataset_loader(input_filepath, normalize=True, 
        debug_size=None, train_split=.8, shuffle=False, timesteps=None,
        num_target_bins=None, balanced_class_loss=False, 
        target_index=None):
    """
    Description:
        - Load a risk dataset from file, optionally normalizing it.

    Args:
        - input_filepath: filepath from which to load
        - noramlize: whether or not to mean center and divide by std dev
        - debug: whether using debug set
        - train_split: fraction of samples used for training
        - shuffle: whether to shuffle the order of the samples

    Returns:
        - data: a dictionary with keys 'x_train', 'y_train', 'x_val', 'y_val'
    """
    infile = h5py.File(input_filepath, 'r')

    # if debugging, use fewer samples
    if debug_size is not None:
        features = infile['risk/features'][:debug_size]
        targets = infile['risk/targets'][:debug_size]
    else:
        features = infile['risk/features'].value
        targets = infile['risk/targets'].value

    # downselect timesteps
    if len(features.shape) > 2 and timesteps is not None:
        features = features[:, -timesteps:,:]
        if features.shape[1] == 1:
            features = np.squeeze(features, axis=1)

    # downselect targets if specified
    if target_index is not None:
        targets = targets[:, target_index, np.newaxis]

    # discretize means break the targets into bins 
    weights = None
    if num_target_bins is not None:
        discretize_targets(targets, num_target_bins)
        if balanced_class_loss:
            weights = get_balanced_class_weights(targets)
        else:
            weights = None

    msg = 'features and targets must be same length: features len: {}\ttargets len: {}'.format(
        len(features), len(targets))
    assert len(features) == len(targets), msg

    # if shuffle then randomly permute order
    if shuffle:
        idxs = np.random.permutation(len(features))
        features = features[idxs]
        targets = targets[idxs]
    
    # separate into train / validation
    num_samples = len(features)
    num_train = int(np.ceil(num_samples * train_split))
    data = {'x_train': features[:num_train],
        'y_train': targets[:num_train],
        'x_val': features[num_train:],
        'y_val': targets[num_train:]}

    if weights is not None:
        data['w_train'] = weights[:num_train]
        data['w_val'] = weights[num_train:]

    # normalize using train statistics
    if normalize:
        data = normalize_features(data)

    # add seeds and batch_idxs if they exist
    data['seeds'] = infile.get('risk/seeds', np.array([]))
    data['batch_idxs'] = infile.get('risk/batch_idxs', np.array([]))

    return data
