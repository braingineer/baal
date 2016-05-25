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
try:
    import cPickle as pickle
except:
    import pickle
try:
    input  = raw_input
except:
    pass
from .base import Igor
from lasagne.regularization import regularize_layer_params
from lasagne.regularization import regularize_network_params
from lasagne.regularization import l2

__all__ = ["GistIgor"]

class GistIgor(Igor):
    """an assistant to clean up my GIST learning experiments

    Args:
        config_file:
            for GIST, this is a very specific format:

            ### data
            training_paths_file : pickle file with paths data
            val_paths_file      : pickle file with paths data
            test_paths_file     : pickle file with paths data ## don't include for now
            vocab_file          : dictionary mapping strings to ids
            save_file           : where to save the parameters

            ### data parameters
            data_mode   "flat"  : yield data as (derivtree,label)
                                  makes pairs out of each node in the path
                        "chain" : yield data as ((derivtree, derivetree,...), label)
                                  this is the actual path

            ### learning parameters
            learning_rate       : rate to update parameters
            num_epochs          : number of passes through the data
            nonlinearity_str   : string choice reflecting hard-coded choices below
            batch_size          : size of data chunk to serve to learning algo
    """
    def __init__(self, *args, **kwargs):
        super(GistIgor, self).__init__(*args, **kwargs)
        if self.rapid_cycle:
            self.rapid_on()
        elif self.hp_on:
            self.hpsearch_on()

    def initialize(self):
        self.logger.info("Starting to initialize")
        train_file = "{}/{}".format(self.data_dir, self.training_paths_file)
        with open(train_file) as fp:
            self.train_data = pickle.load(fp)
        self.logger.info("Training data loaded; {} entries".format(len(self.train_data)))

        val_file = "{}/{}".format(self.data_dir, self.val_paths_file)
        with open(val_file) as fp:
            self.val_data = pickle.load(fp)

        self.logger.info("Val data loaded; {} entries".format(len(self.val_data)))

        #vocab_file = "{}/{}".format(self.data_dir,self.vocab_file)
        #self.vocab = Vocabulary.load(vocab_file)

        #self.logger.info("Vocab loaded; {} entries".format(len(self.vocab)))

    def report(self):
        if len(self.observations) == 0:
            if self.verbose: raise Exception("I can't report; failing loudly")
            elif self.logger: self.logger.warning("Can't report. Failing silently")
            else: return
        obs_time, obs = self.observations[-1]
        epoch, train_loss, train_acc5, train_acc8, val_loss, val_acc5, val_acc8 = obs
        if self.logger:
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("\tTraining Loss: {}".format(train_loss))
            self.logger.info("\tTraining Accuracy@0.5: {}".format(train_acc5))
            self.logger.info("\tTraining Accuracy@0.8: {}".format(train_acc8))
            self.logger.info("\tValidation Loss: {}".format(val_loss))
            self.logger.info("\tValidation Accuracy@0.5: {}: ".format(val_acc5))
            self.logger.info("\tValidation Accuracy@0.8: {}".format(val_acc8))
            self.logger.info("\tLoss Ratio (Val/Train): {}".format(val_loss/train_loss))
        if self.verbose:
            print("Epoch {}".format(epoch))
            print("\tTraining Loss: {}".format(train_loss))
            print("\tValidation Loss: {}".format(val_loss))
            print("\tLoss Ratio (Val/Train): {}".format(val_loss/train_loss))

        self.scribe.record(epoch, type="epoch")
        self.scribe.record(train_loss, type="training loss")
        self.scribe.record(train_acc5, type="training acc5")
        self.scribe.record(train_acc8, type="training acc8")
        self.scribe.record(val_loss, type="validation loss")
        self.scribe.record(val_acc5, type="validation acc5")
        self.scribe.record(val_acc8, type="validation acc8")


    def end_experiment(self, params):
        params = [p.get_value() for p in params]
        super(GistIgor, self).end_experiment()
        loc = "{}/{}".format(self.param_location, self.save_file)
        with open(loc, 'wb') as fp:
            pickle.dump(params, fp)

    def checkpoint(self, params, prefix, epoch=0):
        params = [p.get_value() for p in params]
        if self.checkpoint_limit and epoch % self.checkpoint_limit == 0:
            loc = "{}/{}{}".format(self.param_location, prefix, self.save_file)
            with open(loc, 'wb') as fp:
                pickle.dump(params, fp)

    def rapid_on(self):
        self.training_paths_file = self.rapid_train
        self.val_paths_file = self.rapid_val
        self.num_epochs = self.rapid_epochs
        self.checkpoint_limit = self.rapid_checkpoint

    def hpsearch_on(self):
        print("Hypersearch turning on")
        self.training_paths_file = self.hp_train
        self.val_paths_file = self.hp_val
        self.num_epochs = self.hp_epochs
        self.checkpoint_limit = self.hp_checkpoint

    @property
    def epoch_iter(self):
        for e_i in xrange(self.num_epochs):
            yield e_i

    @property
    def num_training_batches(self):
        if self.train_data:
            return len(self.train_data) // self.batch_size
        return 0

    @property
    def num_val_batches(self):
        if self.val_data:
            return len(self.val_data) // self.batch_size
        return 0

    @property
    def num_test_batches(self):
        if self.test_data:
            return len(self.test_data) // self.batch_size
        return 0

    @property
    def feature_size(self):
        return len(self.vocab)

    @property
    def input_size(self):
        return (self.batch_size, self.feature_size)

    def nonlinearity(self, type):
        if "out":
            return getattr(lasagne.nonlinearities, self.nonlinearity_out)
        elif "hid":
            return getattr(lasagne.nonlinearities, self.nonlinearity_hid)

    @property
    def updater(self):
        return getattr(lasagne.updates, self.learning_algorithm)

    def apply_l2(self, layer, network=False):
        if network:
            return regularize_network_params(layer, l2)
        else:
            return regularize_layer_params(layer, l2)


    def _train_iter(self):
        for label, feature_chain in self.train_data:
            for featureset in feature_chain:
                yield (label, featureset)

    @property
    def training_server(self):
        if not self.train_data:
            raise Exception('no data loaded')

        dataiter = self._train_iter()
        while dataiter:
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) == 0:
                raise StopIteration
            X = np.zeros((self.batch_size, self.feature_size), dtype=np.int8)
            y = np.zeros((self.batch_size, 1), dtype=np.int8)
            for i, (label, ids) in enumerate(next_batch):
                X[i, ids] = 1
                y[i] = label

            yield X,y

    def _var_iter(self):
        for label, feature_chain in self.val_data:
            for featureset in feature_chain:
                yield (label, featureset)

    @property
    def val_server(self):
        if not self.val_data:
            raise Exception('no data loaded')

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


