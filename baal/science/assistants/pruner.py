from __future__ import division, print_function
from baal.utils.timer import Timer
from baal.utils import Vocabulary
import yaml
import json
import pickle
import math
import itertools
import lasagne
import baal
import numpy as np
import theano
from . import scribe
try:
    import cPickle as pickle
except:
    import pickle
try:
    input  = raw_input
except:
    pass

from .gist_base import GistIgor



__all__ = ["Pruner"]

class Pruner(GistIgor):
    """
        This version of the assistant will remove duplicate data points

    Explanation:
        Duplicate data points occur because the current learning method is to
        flatten the paths and treat each path node as a classification problem.
        Good paths and bad paths are guaranteed to share some early nodes.
        So, rather than give duplicate training signals, we prune those states.
    """
    @staticmethod
    def filter_in(chain, filterset):
        return [x for x in chain if x in filterset]

    @staticmethod
    def filter_out(chain, filterset):
        return [x for x in chain if x not in filterset]

    def initialize(self):
        super(Pruner, self).initialize()
        good, bad = set(), set()
        for label, feature_chain in self.train_data:
            for featureset in feature_chain:
                if label:
                    good.add(tuple(featureset))
                else:
                    bad.add(tuple(featureset))

        self.pos_train_data = map(list, good)
        self.neg_train_data = map(list, bad - good)
        self.logger.info("{} data in pos train".format(len(self.pos_train_data)))
        self.logger.info("{} data in neg train".format(len(self.neg_train_data)))


        good, bad = set(), set()
        for label, feature_chain in self.val_data:
            for featureset in feature_chain:
                if label:
                    good.add(tuple(featureset))
                else:
                    bad.add(tuple(featureset))

        self.pos_val_data = map(list, good)
        self.neg_val_data = map(list, bad - good)
        self.logger.info("{} data in pos val".format(len(self.pos_val_data)))
        self.logger.info("{} data in neg val".format(len(self.neg_val_data)))


    @property
    def num_training_batches(self):
        return 2 * len(self.pos_train_data) // self.batch_size

    @property
    def num_training_datapoints(self):
        n_good = len(self.pos_train_data)
        n_bad = len(self.neg_train_data)
        return min(n_good, n_bad) * 2

    @property
    def num_val_datapoints(self):
        return len(self.pos_val_data) + len(self.neg_val_data)

    @property
    def num_val_batches(self):
        return  (len(self.pos_val_data) + len(self.neg_val_data)) // self.batch_size

    def _val_iter(self):
        for datum in self.pos_val_data:
            yield 1, datum
        for datum in self.neg_val_data:
            yield 0, datum

    @property
    def val_server(self):
        dataiter = self._val_iter()
        while dataiter:
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) == 0:
                raise StopIteration
            X = np.zeros((self.batch_size, self.feature_size), dtype=np.int8)
            y = np.zeros((self.batch_size,1), dtype=np.int8)
            for i, (label, ids) in enumerate(next_batch):
                X[i, ids] = 1
                y[i] = label

            yield X,y

    def _train_iter(self):
        n_good = len(self.pos_train_data)
        n_bad = len(self.neg_train_data)
        total = min(n_good, n_bad)
        good_indices = np.random.choice(n_good, size=total, replace=False)
        bad_indices = np.random.choice(n_bad, size=total, replace=False)
        for good_index, bad_index in zip(good_indices, bad_indices):
            yield 1, self.pos_train_data[good_index]
            yield 0, self.neg_train_data[bad_index]

    @property
    def training_server(self):
        dataiter = self._train_iter()
        while dataiter:
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) == 0:
                raise StopIteration
            X = np.zeros((self.batch_size, self.feature_size), dtype=np.int8)
            y = np.zeros((self.batch_size,1), dtype=np.int8)
            for i, (label, ids) in enumerate(next_batch):
                X[i, ids] = 1
                y[i] = label

            yield X,y
