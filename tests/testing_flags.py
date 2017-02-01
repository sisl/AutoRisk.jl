
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

from compression.run_compression import FLAGS

FLAGS.snapshot_filepath = '../data/snapshots/test.weights'
FLAGS.summary_dir = '../data/summaries'
FLAGS.hidden_dim = 32
FLAGS.num_hidden_layers = 2
FLAGS.input_dim = 3
FLAGS.output_dim = 2
FLAGS.verbose = False
