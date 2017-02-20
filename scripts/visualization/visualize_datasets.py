import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import os 
import scipy.stats
import sys
# import seaborn as sns


path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset_loaders
import utils

COLORS = ["red", "green", "blue", "purple", "yellow", "orange"]

def scatter_with_best_fit(features, targets, f_idx, t_idx, d_idx):
    """
    Description:
        - scatter a single set of features vs targets

    Args:
        - features: full set of features
        - targets: full set of targets
        - f_idx: index of feature
        - t_idx: index of target
    """
    # show line of best fit
    r, m = -1, -1
    if np.std(features[:, f_idx]) > 1e-8:

        (m, b, r, p_val, std_err) = scipy.stats.stats.linregress(
            features[:, f_idx], targets[:, t_idx])
        plt.plot(features[:, f_idx], m * features[:, f_idx] + b, 
            c=COLORS[d_idx], label='r: {:.4f} m: {:.4f}'.format(r, m))

    # scatter points
    plt.scatter(features[:, f_idx], targets[:, t_idx], alpha=.3)
    return r, m

def visualize_datasets(datasets, dataset_labels, feature_label, 
        target_labels, output_directory):
    """
    Description:
        - Visualize the features vs targets of a set of datasets.

    Args:
        - datasets: list of datasets
        - dataset_labels: names for datasets
        - feature_labels: names for features
        - target_labels: names for targets
        - output_directory: where to save the plots
    """

    # unpack values
    num_datasets = len(datasets)
    feature_sets = [d['x_train'] for d in datasets]
    target_sets = [d['y_train'] for d in datasets]
    num_features = feature_sets[0].shape[-1]
    num_targets = target_sets[0].shape[-1]

    # plot each feature against each target value
    # and also calculate the corresponding corr coeff
    coeff, slopes = [], []
    for f_idx in range(num_features):
        print("feature {}".format(f_idx))
        for t_idx in range(num_targets):
            fig = plt.figure(figsize=(10,10))
            lgds = []
            # plot each dataset as separate graph in figure
            for d_idx in range(num_datasets):

                plt.subplot('{}1{}'.format(num_datasets, int(d_idx + 1)))
                # plt.subplot(int('21{}'.format(d_idx + 1)))
                r, m = scatter_with_best_fit(feature_sets[d_idx], 
                    target_sets[d_idx], f_idx, t_idx, d_idx)
                coeff.append((r, feature_labels[f_idx], 
                    target_labels[t_idx]))
                slopes.append((r, feature_labels[f_idx], 
                    target_labels[t_idx]))
                
                plt.xlabel('{}'.format(feature_labels[f_idx]))
                plt.ylabel('{}'.format(target_labels[t_idx]))
                plt.title('{}: {} vs {}'.format(
                    dataset_labels[d_idx], 
                    feature_labels[f_idx], 
                    target_labels[t_idx]))
                l = plt.legend(bbox_to_anchor=(.8, -0.3), loc=8, borderaxespad=0.)
                
                # if did not scatter best fit line then there will not be 
                # any legend entry, so only add it if it was in fact plotted
                if r != -1 or m != -1:
                    lgds.append(l)

            # save the figure
            output_filepath = os.path.join(output_directory, 
                '{}_vs_{}.png'.format(
                feature_labels[f_idx], 
                target_labels[t_idx]))
            fig.savefig(output_filepath, bbox_extra_artists=lgds, 
                bbox_inches='tight')
            # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
            plt.savefig(output_filepath)
            plt.close()

    for t_idx in range(num_targets):
        fig = plt.figure(1)
        for d_idx in range(num_datasets):
            plt.subplot(int('31{}'.format(d_idx + 1)))
            n, bins, patches = plt.hist(
                target_sets[d_idx][:, t_idx], 50, alpha=0.5)
            plt.title('{}: {}'.format(
                    dataset_labels[d_idx], target_labels[t_idx]))
        output_filepath = os.path.join(
            output_directory, 'new_target_{}.png'.format(
                target_labels[t_idx]))
        print(output_filepath)
        fig.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    # sorting by correlation
    print(sorted(coeff))
    print(sorted(slopes))
    output_filepath = os.path.join(output_directory, 'corr.npz')
    np.savez(output_filepath, coeff=coeff, slopes=slopes)

def compute_feature_target_correlations(dataset):
    # unpack
    x, y = dataset['x_train'], dataset['y_train']
    num_samples, num_features = x.shape
    _, num_targets = y.shape

    # iterate through each feature
    coeffs = np.zeros((num_features, num_targets))
    for fidx in range(num_features):
        # take the coefficients corresponding to the targets
        coeffs[fidx, :] = np.corrcoef(x[:, fidx], y, rowvar=0)[0, 1:]

    return coeffs

def collision_probability_by_behavior(dataset, politeness_idx=72):
    # output collision probability for the different behavior classes
    x, y = dataset['x_train'], dataset['y_train']
    tar_by_beh = collections.defaultdict(list)
    for s_idx in range(len(x)):
        politeness = x[s_idx, politeness_idx]
        target_probs = y[s_idx, :]
        tar_by_beh[politeness].append(target_probs)
    lines = [(p, np.mean(v, axis=0)) for (p,v) in tar_by_beh.items()]
    print("politeness (indicating behavior class) vs target means")
    for line in lines: print(line)

def report_poorly_performing_indices(idxs, data):
    batch_idxs = data['batch_idxs']
    seeds = data['seeds']
    for idx in idxs:
        # you could biject, but let's keep it simple
        for i, b in enumerate(batch_idxs):
            if b > idx:
                break
        seed = seeds[i]
        if i > 0:
            veh_idx = idx - batch_idxs[i - 1] + 1
        else:
            veh_idx = idx + 1
        print('seed: {}\tveh idx: {}'.format(seed, veh_idx))
        print('seed num veh: {}'.format(batch_idxs[i] - batch_idxs[i-1]))

def compare_dataset_targets_pairwise(d1, d2, target_labels):
    # truncate to the same number of samples
    y1, y2 = d1['y_train'], d2['y_train']
    num_samples = min(len(y1), len(y2))
    y1, y2 = y1[:num_samples], y2[:num_samples]

    # brier score by target
    brier = ((y1 - y2) ** 2).mean(axis=0)
    print('\nbrier scores: {}\n'.format(brier))

    # hypothesis test by target
    # null hypothesis: they are the same
    # alternative hypothesis: they are different
    y1_means, y2_means = y1.mean(axis=0), y2.mean(axis=0)
    p_hat = (y1_means + y2_means) / 2.
    z = np.abs((y1_means - y2_means) / np.sqrt(
        p_hat * (1 - p_hat) * (2 / num_samples)))
    z_alpha = 1.96
    for t, zt in enumerate(z):
        reject = False
        if zt > z_alpha:
            reject = True
        print('target: {}\t reject: {}\tz: {}'.format(
            target_labels[t], reject, zt))
    print('\n')

    # where did it perform poorly
    num_poor_samples = 50
    poor_idxs = np.argsort(np.sum(np.abs(y1 - y2), axis=1))
    for j, idx in enumerate(reversed(poor_idxs)):
        report_poorly_performing_indices([idx], d1)
        print('y1: {}\ny2: {}\n\n'.format(y1[idx], y2[idx]))
        if j > num_poor_samples:
            break

def sort_scenario_seeds_by_target(dataset, output_directory):
    # sort the scenario seeds by target probabilities
    x, y = dataset['x_train'], dataset['y_train']
    seeds, batch_idxs = dataset['seeds'], dataset['batch_idxs']

    # if already computed then just load
    metadata_filepath = os.path.join(output_directory, 'metadata.npz')
    if os.path.exists(metadata_filepath):
        metadata = np.load(metadata_filepath)
        sorted_seeds = metadata['sorted_seeds']
        sorted_means = metadata['sorted_means']
        sorted_bss = metadata['sorted_bss']
    # otherwise compute means by scenario batch and sort along with seeds
    else:
        mean_targets = []
        prev_bidx = 0
        for (i, bidx) in enumerate(batch_idxs):
            ys = y[prev_bidx:bidx]
            bs = bidx - prev_bidx
            # mean_seed = (tuple(np.mean(ys, axis=0)), seeds[i], bs)
            mean_seed = (np.mean(np.mean(ys, axis=0)), seeds[i], bs)
            mean_targets.append(mean_seed)
            prev_bidx = bidx

        sorted_means = sorted(mean_targets, reverse=True)
        sorted_seeds = [s for (_, s, _) in sorted_means]
        sorted_bss = [bs for (_, _, bs) in sorted_means]
        sorted_means = [m for (m, _, _) in sorted_means]

    # display some info
    print(sorted_seeds[:100])
    print(sorted_bss[:100])
    print(sorted_means[:10])

    plt.scatter(range(len(sorted_bss)), sorted_bss, alpha=.5)
    plt.show()
    plt.scatter(range(len(sorted_seeds)), sorted_seeds, alpha=.5)
    plt.show()
    original = sorted(list(zip(sorted_seeds, sorted_means)))
    means = np.array([m for (s,m) in original])
    seeds = [s for (s,m) in original]
    plt.scatter(seeds, means[:, 0], alpha=.5)
    plt.show()

    # save if have't already
    if not os.path.exists(metadata_filepath):
        np.savez(metadata_filepath, sorted_seeds=sorted_seeds, 
            sorted_means=sorted_means, sorted_bss=sorted_bss)

def compare_dataset_features(d1, d2, l1, l2):
    f1 = d1['x_train'][0]
    f2 = d2['x_train'][0]

    map_1 = {lv1:fv1 for (fv1,lv1) in zip(f1, l1)}
    map_2 = {lv2:fv2 for (fv2,lv2) in zip(f2, l2)}

    k1 = set(map_1.keys())
    k2 = set(map_2.keys())

    diff = k1.symmetric_difference(k2)
    print('different keys: {}\nlen: {}'.format(sorted(diff), len(diff)))

    same = k1.intersection(k2)
    for k in l2:
        if k in k1 and k in k2:
            print(k)
            print(map_1[k])
            print(map_2[k])
            input()


def display_target_info(datasets, target_labels, dataset_labels, 
        output_directory):
    # basic stats
    means = [np.mean(d['y_train'], axis=0) for d in datasets]
    print(means)
    stds = [np.std(d['y_train'], axis=0) for d in datasets]
    maxes = [np.max(d['y_train'], axis=0) for d in datasets]
    num_samples = [len(d['y_train']) for d in datasets]
    lines = ["{} # samples: {}\ttarget means: {}\tstds: {}\tmax: {}\n".format(
        l, ns, m, s, mx) for (l, ns, m, s, mx) in zip(
        dataset_labels, num_samples, means, stds, maxes)]
    print(''.join(lines))

    # correlations
    print('target correlation matrix: ')
    print(np.corrcoef(datasets[-1]['y_train'], rowvar=0))

    # plot targets for each dataset
    # normalize across means
    means = means / np.max(means, axis=0)
    num_datasets, target_dim = means.shape
    width = 0.2
    indices = np.arange(target_dim)
    rects = []
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    for i in range(num_datasets):
        rect = ax.bar(indices + width * i, means[i,:], width, color=COLORS[i], 
            alpha=.9, edgecolor="black")
        rects.append(rect)
    
    # change in target across datasets
    ax.set_ylabel('pr(target)')
    ax.set_title('pr(target) across datasets')
    print(target_labels)
    plt.xticks(indices, target_labels)
    ax.legend(([rect[0] for rect in rects]), dataset_labels, bbox_to_anchor=(.75, .98))
    output_filepath = os.path.join(output_directory, 'target_comparison.png')
    plt.savefig(output_filepath)

    # histogram of target values
    _, num_targets = datasets[0]['y_train'].shape
    num_datasets = len(datasets)
    target_sets = [d['y_train'] for d in datasets]
    for t_idx in range(num_targets):
        # fig = plt.figure(1)
        for d_idx in range(num_datasets):
            plt.subplot(int('{}1{}'.format(num_datasets, d_idx + 1)))
            n, bins, patches = plt.hist(
                target_sets[d_idx][:, t_idx][target_sets[d_idx][:, t_idx] > 0], 
                50, alpha=0.5)
            plt.title('{}: {}'.format(
                    dataset_labels[d_idx], target_labels[t_idx]))
        output_filepath = os.path.join(
            output_directory, 'new_target_{}.png'.format(
                target_labels[t_idx]))
        print(output_filepath)
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()

    # for tidx in range(target_dim):
        


    #     plt.scatter(np.arange(num_datasets) * 5, means[:, tidx], c='green', alpha=.5)
    #     plt.title(target_labels[tidx])
    #     plt.xlabel('seconds')
    #     plt.ylabel('pr(target)')
    #     output_filepath = os.path.join(output_directory, 
    #         '{}_across_datasets.png'.format(target_labels[tidx]))
    #     plt.savefig(output_filepath)
    #     plt.close()

if __name__ == '__main__':
    # labels stored in external csv files 
    feature_labels_filepath = '../../data/datasets/features.csv'
    target_labels_filepath = '../../data/datasets/targets.csv'

    # load in labels
    feature_labels = utils.load_labels(feature_labels_filepath)
    target_labels = utils.load_labels(target_labels_filepath)

    # the dataset filepaths to visualize along with labels
    input_filepaths = [
        '../../data/datasets/2_19/test.h5',
        # '../../data/datasets/2_19/risk_.1_sec_.5_sec_features.h5',
        # '../../data/datasets/2_13/risk_learned.h5',
        # '../../data/datasets/2_13/risk_.1_sec_lidar_features.h5',
        '../../data/datasets/2_13/risk_.1_sec_regular_features.h5'
        # '../../data/datasets/2_4/risk_.1_sec_multi.h5',
        # '../../data/datasets/2_4/risk_.1_sec.h5',
        # '../../data/datasets/1_30/risk_10_sec.h5',
        # '../../data/datasets/1_30/risk_10_sec_bootstrap.h5',
        # '../../data/datasets/1_22/risk_5_second_fix_complete.h5',
        # '../../data/datasets/1_30/risk_5_sec_delay.h5',
        # '../../data/datasets/1_22/risk_5_second_no_idm_decel.h5',
        # '../../data/datasets/1_22/bootstrap_10_sec.h5',
        # '../../data/datasets/1_22/risk_10_second_full.h5',
        # '../../data/datasets/1_22/risk_5_second_complete.h5',
        # '../../data/datasets/1_8/risk_20_second_thresh_5_second.h5',
        # '../../data/datasets/1_8/risk_5_second.h5',
        # '../../data/datasets/1_8/risk_10_second.h5',
        # '../../data/datasets/1_1/risk_5_second.h5',
        # '../../data/datasets/1_1/risk_10_second.h5',
        # '../../data/datasets/1_1/risk_24.h5', 
        # '../../data/datasets/1_1/risk_25.h5', 
        # '../../data/datasets/1_1/risk_26.h5', 
        # '../../data/datasets/11_27/risk_16.h5',
        # '../../data/datasets/12_5/risk_halftime.h5',
        # '../../data/datasets/11_27/risk_no_noise.h5',
        # '../../data/datasets/11_14/risk_ngsim.h5'
        # '../../data/datasets/12_11/risk_18.h5'
    ]
    dataset_labels = [
        'test'
        # 'multi',
        # 'lidar',
        'regular'
        # 'risk_10_sec',
        # 'risk_10_sec_bootstrap',
        # 'risk_5_second',
        # 'risk_5_second_delay',
        # 'risk_5_second_no_idm_decel',
        # '10_sec_bootstrap',
        # '10_sec',
        # '5_sec',
        # '10_sec',
        # '10_sec_thresh',
        # '20_sec_thresh',
        # '5_sec',
        # '10_sec',
        # '5_sec_3_sec_burn_in',
        # '10_sec_3_sec_burn_in',
        # '20_sec'
    ]

    # input_filenames = ['iter_{}.h5'.format(i) for i in range(10)]
    # basedir = '/Users/wulfebw/Desktop/temp_risk/datasets'
    # input_filepaths = [os.path.join(basedir, f) for f in input_filenames]
    # dataset_labels = ['{}_seconds'.format(i * 5) for i in range(1, 11)]
    
    # check for / create output directory
    base_directory = '../../data/visualizations/'
    output_directory = os.path.join(
        base_directory, os.path.split(
            input_filepaths[0])[-1]).replace('.h5', '')
    # output_directory = os.path.join(base_directory, 'risk_beh')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # load in each dataset
    debug_size = 1000000
    datasets = [dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=False, train_split=1., 
        debug_size=debug_size, normalize=False, timesteps=1) for 
        input_filepath in input_filepaths]

    # display basic info about the targets
    # display_target_info(datasets, target_labels, dataset_labels, output_directory)

    # ## analyze behavior
    # for i, dataset in enumerate(datasets):
    #     print(dataset_labels[i])
    #     collision_probability_by_behavior(dataset)

    # compute feature target correlations and write to file
    # coeffs = compute_feature_target_correlations(datasets[-1])
    # output_filepath = os.path.join(output_directory, 'coeff.csv')
    # utils.write_to_csv(output_filepath, coeffs, feature_labels, target_labels)

    # # visualize the datasets
    # visualize_datasets(datasets, dataset_labels, feature_labels, 
    #     target_labels, output_directory)

    # output_directory = os.path.join(base_directory, dataset_labels[-1])
    # sort_scenario_seeds_by_target(datasets[-1], output_directory)

    # # compare two datasets
    # compare_dataset_targets_pairwise(
    #     datasets[0], datasets[1], target_labels)

    feature_labels_filepath = '../../data/datasets/features_multi.csv'
    feature_labels_multi = utils.load_labels(feature_labels_filepath)
    compare_dataset_features(datasets[0], datasets[1], feature_labels_multi, feature_labels)
