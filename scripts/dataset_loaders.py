
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
    mean = np.mean(data['x_train'], axis=0)
    data['x_train'] = (data['x_train'] - mean)

    # compute the standard deviation after mean subtraction
    std = np.std(data['x_train'], axis=0)

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

def risk_dataset_loader(input_filepath, normalize=True, 
        debug_size=None, train_split=.8, shuffle=False):
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
    num_train = int(num_samples * train_split)
    data = {'x_train': features[:num_train],
        'y_train': targets[:num_train],
        'x_val': features[num_train:],
        'y_val': targets[num_train:]}

    # normalize using train statistics
    if normalize:
        data = normalize_features(data)

    # add seeds and batch_idxs if they exist
    data['seeds'] = infile.get('risk/seeds', np.array([]))
    data['batch_idxs'] = infile.get('risk/batch_idxs', np.array([]))

    return data
