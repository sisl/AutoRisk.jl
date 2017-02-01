
import collections
import heapq
import numpy as np
import random
import sys

def find_subcutpoints(probs, cutpoints, partition_prob):
    num_elements = len(probs)
    cur_prob, cutpoints, skip, residual = 0, [], False, 0
    for i in range(num_elements - 1):

        # if the previous iteration added two elements, then skip this one
        if skip: 
            skip = False
            continue

        cur_prob += probs[i]
        next_prob = cur_prob + probs[i + 1]

        if next_prob > partition_prob:

            # have extra probability mass to spare, cut after the next value
            if residual > 0:
                cutpoints.append(i + 2)
                residual += partition_prob - next_prob
                skip = True

            # otherwise, lacking probability mass, cut before the next value
            else:
                cutpoints.append(i + 1)
                residual += partition_prob - cur_prob

            cur_prob = 0

    # check if the last element merits adding a cutpoint
    if cur_prob + probs[-1] >= partition_prob:
        cutpoints.append(num_elements)

    return cutpoints

def find_cutpoints(probs, num_partitions):
    probs = np.asarray(probs, dtype=np.float128)
    partition_prob = sum(probs) / float(num_partitions)
    cutpoints = find_subcutpoints(probs, num_partitions, partition_prob)

    while len(cutpoints) < num_partitions:

        # locate next cutpoint covering more than one element
        prev_c = 0
        for i, c in enumerate(cutpoints):
            if c - prev_c > 1:
                break
            prev_c = c

        # find cutpoints for probs following the cutpoint
        subcutpoints = find_subcutpoints(
            probs[c:], num_partitions - (i + 1), partition_prob)
        base = c - 1
        subcutpoints = [base + sc for sc in subcutpoints]

        # the new cutpoints will start with the previous cutpoints, except that
        # one cutpoint will be reduced by an element
        if i == 0:
            cutpoints = [cutpoints[0] - 1] + subcutpoints
        else:
            cutpoints = cutpoints[:i] + [c - 1] + subcutpoints

    cutpoints[-1] = len(probs)
    return cutpoints

def siftdown(heap, startpos, pos, idx_dict):
    orig = idx_dict[pos]
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            if parentpos in idx_dict:
                idx_dict[parentpos] = pos
            pos = parentpos
            continue
        break
    heap[pos] = newitem
    idx_dict[orig] = pos

def siftup(heap, pos, idx_dict):
    endpos = len(heap)
    orig = idx_dict[pos]
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        if childpos in idx_dict:
                idx_dict[childpos] = pos
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    idx_dict[orig] = pos
    siftdown(heap, startpos, pos, idx_dict)

class PrioritizedDataset(object):

    def __init__(self, data, flags):
        self.data = data
        self.flags = flags
        self.sample_count = 0
        self.heap = []

        for x, y in zip(data['x_train'], data['y_train']):
            self.store((x, y))

        self.repartition(flags.batch_size, flags.priority_alpha, 
            flags.priority_beta)

        # compute batch information
        for split in ['train', 'val']:
            num_samples = len(data['x_{}'.format(split)])
            num_batches = int(num_samples / self.flags.batch_size)

            # if num_samples not divisible by batch_size, then 
            # simply add an additional batch, which will be addressed
            # in next_batch using python indexing past the end of a container
            if num_samples % self.flags.batch_size != 0:
                num_batches += 1

            if split == 'train':
                self.num_train_batches = num_batches
            else:
                self.num_val_batches = num_batches

        # allocate space for batch a single time
        self.x_train = np.empty((self.flags.batch_size, self.flags.input_dim), 
                        dtype=np.float32)
        self.y_train = np.empty((self.flags.batch_size, self.flags.output_dim), 
                        dtype=np.float32)

    def store(self, sample, priority=-1000):
        self.sample_count += 1
        # new samples are added with highest priority, which is the smallest 
        # value since this is a min heap. We also want newer samples at the top
        # so add the negative sample count to differentiate between equal priorities
        heapq.heappush(self.heap, (priority, -self.sample_count, sample))

    def repartition(self, batch_size, alpha, beta):
        assert alpha >= 0
        assert beta >= 0 and beta <= 1

        # sort heap
        self.heap.sort()
        heap_size = len(self.heap)

        # recompute cutpoints
        # the probability for each sample to be selected is inversely 
        # proportional to its rank. It is purposefully independent of the 
        # actual priority value, and is scaled by exponentiating by alpha.
        probs = np.arange(1., heap_size + 1) ** -alpha
        total_prob = sum(probs)
        probs /= total_prob
        cutpoints = find_cutpoints(probs, batch_size)

        # store the partition sizes, cutpoints, and partition importance weights
        self.partition_sizes = np.array([cutpoints[0]] + [b - a
            for (a, b) in zip(cutpoints, cutpoints[1:])], dtype=float)
        self.cutpoints = np.asarray(cutpoints)

        # the weights are computed to correct for the bias of oversampling 
        # beta varies between 0 (no correction) and 1 (full correction)
        # there are two options:
        # 1. use the 'true' above-computed probabilities for weighting
        # 2. use the actual sampling probabilities (i.e., the approximate
        # probabilities actually used)
        # we go with the second option
        self.importance_weights = (heap_size * 
            (self.partition_sizes * batch_size) ** -1) ** -beta
        self.importance_weights /= np.max(self.importance_weights)
        self.importance_weights = self.importance_weights.reshape(-1, 1)

        # need to store indices into the heap when sampling to update priority
        self.heap_idxs = np.empty(batch_size, dtype=np.int64)

    def sample_batch(self):
        # select a sample_idx uniformly at random from each partition 
        for idx, (prev_c, c) in enumerate(zip(
                np.hstack(([0], self.cutpoints)), self.cutpoints)):
            heap_idx = random.randint(prev_c, c - 1)
            self.heap_idxs[idx] = heap_idx

            # add sample to batch
            (x, y) = self.heap[heap_idx][2]
            self.x_train[idx] = x
            self.y_train[idx] = y

    def next_batch(self, validation=False):

        if validation:
            x, y = self.data['x_val'], self.data['y_val']
            idxs = np.random.permutation(len(x))
            x = x[idxs]
            y = y[idxs]

            # yield data in batches
            for bidx in range(self.num_val_batches):
                # compute start and end indices
                start = bidx * self.flags.batch_size
                end = (bidx + 1) * self.flags.batch_size

                yield x[start:end], y[start:end]

        else:
            for bidx in range(self.num_train_batches):
                self.sample_batch()
                yield (self.x_train, self.y_train, self.importance_weights)

    def update_priorities(self, priorities):

        # iterate through the heap indices previously sampled, updating their 
        # priorities
        idx_dict = collections.defaultdict(int)
        idx_dict.update({i:i for i in self.heap_idxs})
        for (orig_hidx, p) in zip(self.heap_idxs, priorities):
            # get the possibly-updated index in the heap
            hidx = idx_dict[orig_hidx]

            # increase the priority of the key
            if p[0] < self.heap[hidx][0]:
                self.heap[hidx] = (p[0], self.heap[hidx][1], self.heap[hidx][2])
                siftdown(self.heap, 0, hidx, idx_dict)

            # decrease the priority of the key
            else:
                self.heap[hidx] = (p[0], self.heap[hidx][1], self.heap[hidx][2])
                siftup(self.heap, hidx, idx_dict)
        