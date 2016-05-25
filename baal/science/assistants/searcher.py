"""
Assist beam searchers
"""
from __future__ import print_function, division
import yaml
from .attender import Attender
import baal
from os.path import join
import pickle
from baal.nlp.structures import GenerationTree
from baal.utils import VocabManager
import pickle

class Searcher(Attender):
    def __init__(self, yaml_config):
        self.__dict__.update(yaml_config)
        self.base_path = join(baal.PATH, "science", "gist")
        self.initialize()

    def initialize(self):
        F = lambda v: join(self.vocab_dir, v)
        self.vocman = VocabManager(True)
        self.vocman.add("tree_feats", filename=F(self.tree_feature_vocab))
        self.vocman.add("attach_feats", filename=F(self.attach_feature_vocab))
        self.vocman.add("tree_index", filename=F(self.tree_index_filename))

        self.tree_feature_size = len(self.vocman.tree_feats)
        self.attachment_feature_size = len(self.vocman.attach_feats)

        with open(join(self.base_path, self.grammar_file)) as fp:
            G = [GenerationTree.from_bracketed(g) for g in pickle.load(fp)]
        head_lookup = {}
        for g in G:
            head_lookup.setdefault(g.E.head, []).append(g)
        self.vocman.head_lookup = head_lookup

        #self.max_num_elemtrees = self.subsets.shape[2]

        #self.train_data, mtrain = self.flatten_data(self.train_data)
        #self.num_training_datapoints = len(self.train_data)
        #self.val_data, mval = self.flatten_data(self.val_data)
        #self.num_val_datapoints = len(self.val_data)
        #self.max_num_attachments = max(mtrain,mval)

    @classmethod
    def from_file(cls, filename):
        with open(filename) as fp:
            return cls(yaml.load(fp))