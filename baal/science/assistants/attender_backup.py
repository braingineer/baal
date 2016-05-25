"""Assistant for Generating Trees with Attention and GIST

Data Description:
    data:
        type: dict
        content:    {surface_form:
                      [<path>, <path>, ...],
                     ...}
    path:
        type: dict
        content: {'dtree': tree_features,
                  'attach_points': attach_point_features,
                  'attach_target': local_point_index,
                  'tree_target': tree_index,
                  'attach_type': (type_index, pos_index)}
    tree_features:
        type: list
        content: [tree_feature_index, tree_feature_index, ...]
        note: tree_feature_index is an alais for: tfeat_index or tree_feat_index
    attach_point_features:
        type: list
        content: [[attach_feature_index, ...],
                  [attach_feature_index, ...],
                  ...]
        note: attach_feature_index is an alias for: afeat_index or attach_feat_index
    local_point_index:
        type: int
        content: index to the attach_point_features array
        note: the only index that differs in that it references an array
    X_index:
        type: int
        content: vocabulary encoding of X;
                 see X_vocab
    X_vocab:
        type: baal.utils.Vocabulary
        content: the string-to-integer mapping for X domain;
        X:
            tree_index := encodes the elementary trees
            tree_feature := encodes the features of trees
            attach_feature := encodes the features of attachment points
            type := encodes the tree type (currently: ins for insert and sub for substitution)
            pos := encodes the part of speech of the root node (the tree operation node)

"""
from __future__ import division, print_function
import yaml
import json
import pickle
import math
import itertools
import lasagne
import baal
import numpy as np
import theano
import os
from baal.utils.timer import Timer
from baal.utils import Vocabulary
from baal.utils import VocabManager
from .gist_base import GistIgor

try:
    import cPickle as pickle
except:
    import pickle
try:
    input  = raw_input
except:
    pass

class utils:
    #log = loggers.duallog("decode")

    @staticmethod
    def unzip(arr, i):
        assert isinstance(arr, list)
        return [x[i] for x in arr]

    @staticmethod
    def indices(mat):
        return np.arange(len(mat)).reshape(len(mat),1), mat

    @staticmethod
    def make_data_matrices(ref):
        return (np.zeros(ref.dtree_shape, dtype=theano.config.floatX),
                np.zeros(ref.attach_shape, dtype=theano.config.floatX),
                np.zeros(ref.etree_shape, dtype=theano.config.floatX),
                np.zeros(ref.next_dtree_shape, dtype=theano.config.floatX))

    @staticmethod
    def parse2predicates(input_parse):
        baal.utils.hlf.reset()
        entry = Entry.make(bracketed_string=input_parse)
        tree, surface = entry.tree, entry.get_lexical()
        tree_enrichment.populate_annotations(tree)
        tree_enrichment.recursive_spine_fix(tree)
        _, new_addressbook = tree.clone()
        predicates = simple_hlf.from_addressbook(new_addressbook, preprocess=True)
        return predicates, surface

    @staticmethod
    def pad_to(arr, sh):
        pad_width = tuple((0,max(m-n,0)) for n,m in zip(arr.shape, sh))
        return np.pad(arr, pad_width=pad_width, 
                           mode='constant', constant_values=0)

    @staticmethod
    def pad_right(arr, sh):
        pad_dims = tuple((0,max(sh_n-arr_n,0)) for arr_n, sh_n in zip(arr.shape[1:], sh))
        pad_width = ((0,0),) + pad_dims
        return np.pad(arr, pad_width=pad_width, 
                           mode='constant', constant_values=0)

    @staticmethod
    def vstack(arrs, ndim=2, mask=None):
        """ a ton of ifs... but a ton of edge cases. should work for a range of cases """
        out = None
        for arr in arrs:
            if arr is None:
                continue
            if isinstance(arr, list):
                arr = np.array(arr)
            if arr.ndim < ndim:
                arr = arr.reshape((1,)*(ndim-arr.ndim)+arr.shape)
            if out is None:
                out = arr
                mask = mask if mask is not None else np.ones_like(arr)
            else:
                sh = tuple(max(i,j) for i,j in zip(out.shape[1:],arr.shape[1:]))
                out = np.vstack((utils.pad_right(out,sh), 
                                 utils.pad_right(arr,sh)))
                mask = np.vstack((utils.pad_right(mask,sh), 
                                  utils.pad_right(np.ones_like(arr),sh)))
        return out, mask
        

    @staticmethod
    def sub_mask(mask):
        return [slice(-1)] + [slice(None, i) for i in mask.shape]


class Attender(GistIgor):
    """This version will handle the attention-based model

    It will generate data that includes attachment points
    """
    def initialize(self):
        """In GistIgor, the data is loaded from pickles into:
            self.train_data
            self.val_data
            self.vocab (superfluous?)
        """
        super(Attender, self).initialize()
        # shape: (num_type, num_pos, num_elemtrees, 2)
        # it is two because an elem tree can only have 2 base features:
        #       head word and template.
        F = lambda v: os.path.join(self.vocab_dir, v)
        self.subsets = np.load(F(self.attention_subsets_file))
        noise_file = self.__dict__[self.selected_noise]
        self.nce_noise = np.load(F(noise_file))
        self.vocman = VocabManager()
        self.vocman.add("tree_feats", filename=F(self.tree_feature_vocab))
        self.vocman.add("attach_feats", filename=F(self.attach_feature_vocab))
        self.vocman.add("head_map", filename=F(self.head_map_filename))

        self.tree_feature_size = len(self.vocman.tree_feats)
        self.attachment_feature_size = len(self.vocman.attach_feats)
        self.max_num_elemtrees = self.subsets.shape[2]

        self.train_data, mtrain = self.flatten_data(self.train_data)
        self.num_training_datapoints = len(self.train_data)
        self.val_data, mval = self.flatten_data(self.val_data)
        self.num_val_datapoints = len(self.val_data)
        self.max_num_attachments = max(mtrain,mval)

        #try:
        #    train_max_att = max(len(node['attach_points']) for paths in self.train_data.values()
        #                                                              for path in paths for node in path)#

        #    val_max_att = max((len(node['attach_points']) for paths in self.val_data.values()
        #                                           for path in paths for node in path))
        #    self.max_num_attachments = max(train_max_att, val_max_att)
        #    #assert self.max_num_attachments >= val_max_att
        #except Exception as e:
        #    import pdb
        #    pdb.set_trace()


        #self.num_training_datapoints = sum(len(path) for pathset in self.train_data.values()
        #                                             for path in pathset)
        #self.num_val_datapoints = sum(len(path) for pathset in self.val_data.values()
        #                                        for path in pathset)

    def flatten_data(self, data):
        nodes = []
        max_attachments = 0.
        bad_datums = 0.
        for sf,pathset in data.items():
            for path in pathset:
                for node in path:
                    if len(node.keys()) == 5:
                        node['attach_type'] = node['tree_target']
                        node['tree_target'] = node['attach_head']
                        node['attach_head'] = None
                    node['attach_points'] = [[v for v in vs if v]
                                              for vs in node['attach_points']]
                    if any(len(v) == 0 for v in node['attach_points']):
                        raise Exception("Bad attachment point features; shouldnt be happening")

                    node['dtree'] = [v for v in node['dtree'] if v]
                    if len(node['dtree']) == 0:
                        raise Exception("Bad tree features. Shouldn't be happening")

                    if node['tree_target'] is None:
                        bad_datums += 1
                        continue

                    nodes.append(node)
                    max_attachments = max(max_attachments, len(node['attach_points']))
        print("Threw away {} data points because of unseen target trees.".format(bad_datums))
        print("Left with {} data points".format(len(nodes)))
        return nodes, max_attachments

    @property
    def default_Fout(self):
        #return lambda x: theano.tensor.maximum(x, 0.)
        return lasagne.nonlinearities.LeakyRectify(0.1)
        #return getattr(lasagne.nonlinearities, self.default_nonlinearity)

    @property
    def default_initializer(self):
        glo = lasagne.init.GlorotUniform(gain='relu')
        #glo.set_positive()
        return glo

    @property
    def linear_initializer(self):
        return lasagne.init.GlorotUniform()

    @property
    def val_server(self):
        return self._compact_server(self._iter(self.val_data))
        return self._server(self._iter(self.val_data))

    @property
    def training_server(self):
        return self._compact_server(self._iter(self.train_data))
        return self._server(self._iter(self.train_data))


    def debug(self, *args, **kwargs):
        """ this will let me do a whole bunch of neat stuff to the debug statements """
        # for now:
        self.logger.debug("  |  ".join(args))

    def make_nce_data(self, attach_type, tree_index=None, tree_feats=None, alternate=False):
        """get the trees resulting from the noise distribution

        Note: this argument is agnostic to the noise distribution.
              it should be passed in as the nce_sampler filename rather than
              be any sort of algorithmic choice.
              The construction happens during the feature generation.
              There, it constructs the count tensor and normalizes across the count dimension.
              It also constructs the flattened empirical unigram, which
        """
        tree_distribution = self.nce_noise[attach_type]
        if self.nce_size > len(tree_distribution.nonzero()[0]) or alternate:
            tree_distribution = self.nce_noise.sum(axis=(0,1))
            tree_distribution /= tree_distribution.sum()
        sample_space = np.arange(self.max_num_elemtrees)
        sample_space = np.delete(sample_space, tree_index)
        sample_distribution = np.delete(tree_distribution, tree_index)
        sample_distribution /= sample_distribution.sum()
        samples = np.random.choice(sample_space,
                                   size=self.nce_size,
                                   replace=False,
                                   p=sample_distribution)
        all_trees = np.concatenate((samples, np.array([tree_index]))).astype(np.int32)
        target_ind = np.random.randint(0, self.nce_size)
        all_trees[-1], all_trees[target_ind] = all_trees[target_ind], all_trees[-1]
        all_noise = tree_distribution[all_trees]
        if all_noise.sum() > 1.0:
            print("THIS IS IN ERROR")
            import pdb
            pdb.set_trace()
        return self.subsets[attach_type][all_trees,:].astype(np.int32), target_ind, all_noise


    def _iter(self, data):
        """interface between data dict and numpy matrix creation
        """
        epoch_size = len(data) // self.batch_size  * self.batch_size
        self.logger.debug("{} dp total across batch sizes of {}".format(epoch_size, self.batch_size))
        for data_i in np.random.choice(len(data),epoch_size, replace=False):
            node = data[data_i]
            dtree_feats = np.array(node['dtree'])
            attach_feats = [x for x in node['attach_points']]
            attach_target = node['attach_target']
            tree_ind = node['tree_target']
            ap_head = node['attach_head']
            if tree_ind is None:
                raise Exception("this shouldn't be happening; encountered bad data")
            atype = node['attach_type']

            #### make the NCE data
            etree_feats, target_id, noise = self.make_nce_data(atype, 
                                                               tree_index=tree_ind,
                                                               alternate=data_i%2)
            #### binding feature stuff
            ## note: implicit assumption that self.nce_size+1 == etree_feats.shape[0]
            binding_feats = np.zeros(self.nce_size+1, dtype=np.int32)
            for etree_i, etree in enumerate(etree_feats):
                etree_head_ind = etree[0]

                key = (ap_head, etree_head_ind, atype)
                if key in self.vocman.head_map:
                    binding_feats[etree_i] = self.vocman.head_map[key]

            #### output
            yield (dtree_feats, attach_feats, attach_target, 
                                etree_feats, target_id, binding_feats, noise)

    def _server(self, dataiter):
        """
            d_tree_in: the derivation trees at time t
            attach_in: the attachments on the derivation trees
            E_target: the indices of the correct elementary trees
            A_target: the indices of the correct attachments
            X_next: the derivation trees that result from possible elem trees
            e_filter: the filter from all elem trees to the subset of possible elem trees

            e_filter and X_next should have same dimensions because:
                E[:, e_filter] = E_filtered              # (h_e, all_e) -> (h_e, e_subset)
                F_align(T, C) * E_filtered = R_E     # (b, h_e) * (h_e, e_subset) -> (b, e_subset)
                Q = R_E + X_next
        """
        dtree_input_shape = (self.batch_size, self.tree_feature_size)
        next_tree_shape = (self.batch_size, self.nce_size+1, self.tree_feature_size)
        attach_input_shape = (self.batch_size, self.max_num_attachments, self.attachment_feature_size)
        etree_target_shape = (self.batch_size, self.nce_size+1)
        attach_target_shape = (self.batch_size, self.max_num_attachments)

        f = lambda x: "{} -> {}".format(x, reduce(lambda a,b:a*b, x))
        self.debug("dtree input shape", f(dtree_input_shape))
        self.debug("attach input shape", f(attach_input_shape))
        self.debug("next tree shape", f(next_tree_shape))
        self.debug("etree target shape", f(etree_target_shape))
        self.debug("attach target shape", f(attach_target_shape))
        while dataiter:
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) < self.batch_size:
                print("End of batch; ({})".format(len(next_batch)))
                raise StopIteration


            # convention: X is input; Y is target; approximating function F(X,Y)
            # negative samplings gives us YNOT.
            deriv_tree_X = np.zeros(dtree_input_shape, dtype=theano.config.floatX)
            attachments_X = np.zeros(attach_input_shape, dtype=theano.config.floatX)

            attachments_Y = np.zeros(attach_target_shape, dtype=theano.config.floatX)

            # the second input route
            elem_trees_X2 = np.zeros(next_tree_shape, dtype=theano.config.floatX)
            deriv_trees_X2 = np.zeros(next_tree_shape, dtype=theano.config.floatX)

            # for computing NCE
            tree_noise = np.zeros(etree_target_shape, dtype=theano.config.floatX)

            elem_trees_Y = np.zeros(etree_target_shape, dtype=theano.config.floatX)


            for i, (dtree_feats, attach_feats, attach_target, etree_feats,
                                 tree_target, binding_feats, noise) in enumerate(next_batch):

                #### inputs
                deriv_tree_X[i, dtree_feats] = 1
                for j, feats in enumerate(attach_feats):
                    attachments_X[i, j, feats] = 1
                #### targets
                elem_trees_Y[i, tree_target] = 1
                attachments_Y[i, attach_target] = 1
                #### nce data for elementary trees
                ### the idea: we broadcast an indexing array (idx)
                ###           across the num_etrees * num_feats column
                ##            this way, we can index the tensor with two arrays
                ##            otherwise, it's super annoying
                ##            it's much easier to have a 1:1 correspondance for points
                ##            even if that means repeating indices in idx (which we do)
                idx = np.arange(etree_feats.shape[0])[:,np.newaxis]
                idx = (idx * np.ones(etree_feats.shape)).flatten().astype(np.int32)
                elem_trees_X2[i, idx, etree_feats.flatten()] = 1
                #### nce data for derivation trees
                ##  the idea: we want the next tree feats to have the current dtree's feats
                ##            plus the feats of the elem tree of choice
                ##            additionally, we need it to have the binding feat
                ##            thing means looking up the head of the attach point
                ##            looking up the head of the elem tree
                ##            checking to see if it matches a feature
                ##            and then including that feature index
                bc_shape = (etree_feats.shape[0],len(dtree_feats))
                dtree_feats = np.broadcast_to(np.array(dtree_feats), bc_shape)
                composed_feats = np.concatenate((etree_feats, dtree_feats), axis=1)
                #composed_feats = feats + dtree_feats + bfeat
                idx = np.arange(composed_feats.shape[0])[:,np.newaxis]
                idx = (idx * np.ones(composed_feats.shape, dtype=np.int32)).flatten()
                deriv_trees_X2[i, idx, composed_feats.flatten()] = 1
                deriv_trees_X2[i, binding_feats.nonzero(), 
                                  binding_feats[binding_feats.nonzero()]] = 1
                #### the noise
                tree_noise[i,:] = noise
            for x in (deriv_tree_X, attachments_X, elem_trees_X2, deriv_trees_X2,
                   attachments_Y, elem_trees_Y,  tree_noise):
                if np.any(np.isnan(x)):
                    raise Exception("FOUND A NAN")
            yield (deriv_tree_X, attachments_X, elem_trees_X2, deriv_trees_X2,
                   attachments_Y, elem_trees_Y,  tree_noise)


    def _compact_server(self, dataiter):
        """
            d_tree_in: the derivation trees at time t
            attach_in: the attachments on the derivation trees
            E_target: the indices of the correct elementary trees
            A_target: the indices of the correct attachments
            X_next: the derivation trees that result from possible elem trees
            e_filter: the filter from all elem trees to the subset of possible elem trees

            e_filter and X_next should have same dimensions because:
                E[:, e_filter] = E_filtered              # (h_e, all_e) -> (h_e, e_subset)
                F_align(T, C) * E_filtered = R_E     # (b, h_e) * (h_e, e_subset) -> (b, e_subset)
                Q = R_E + X_next
        """


        while dataiter:
            dtree_X, dtree_mask = None, None
            attach_X, attach_mask = None, None
            etree_X, etree_mask = None, None
            nexttree_X, nexttree_mask = None, None
            noise_X = None
            etree_Y = None
            attach_Y = None
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) < self.batch_size:
                print("End of batch; ({})".format(len(next_batch)))
                raise StopIteration
            for i, (dtree_feats, attach_feats, attach_target, etree_feats,
                    tree_target, binding_feats, noise) in enumerate(next_batch):

                ### derivation tree input
                dtree_X, dtree_mask = utils.vstack([dtree_X, dtree_feats], 
                                                    mask=dtree_mask)

                ### attachment input
                _feats, _mask =  utils.vstack(attach_feats)
                attach_X, attach_mask = utils.vstack([attach_X, _feats], ndim=3,
                                                     mask=attach_mask)
                attach_mask[utils.sub_mask(_mask)] = _mask

                ### attachment label
                _Y = np.zeros(len(attach_feats))
                _Y[attach_target] = 1
                attach_Y, _ = utils.vstack([attach_Y, _Y])

                ### elementary tree input
                _feats, _mask = utils.vstack(etree_feats)
                etree_X, etree_mask = utils.vstack([etree_X, _feats], ndim=3,
                                                   mask=etree_mask)
                etree_mask[utils.sub_mask(_mask)] = _mask

                ### elementary tree label
                _Y = np.zeros(len(etree_feats))
                _Y[tree_target] = 1
                etree_Y, _ = utils.vstack([etree_Y, _Y])

                
                ### noise
                noise_X, _ = utils.vstack([noise_X, noise])    

                ### next derivation tree
                compose_shape = (etree_feats.shape[0], dtree_feats.shape[0])
                dtree_broadcast = np.broadcast_to(dtree_feats, compose_shape)
                composed = np.concatenate((dtree_broadcast, 
                                           etree_feats, 
                                           binding_feats[:,None]), axis=1)

                ### next derivationt tree; approximating features
                _feats, _mask = utils.vstack(composed)
                _mask[:,-1] = binding_feats
                nexttree_X, nexttree_mask = utils.vstack([nexttree_X, _feats], ndim=3,
                                                         mask=nexttree_mask)
                nexttree_mask[utils.sub_mask(_mask)] = _mask

            import pdb
            #pdb.set_trace()
            yield (dtree_X, dtree_mask[...,None], attach_X, attach_mask[...,None], 
                   attach_Y, etree_X, etree_mask[...,None], etree_Y, nexttree_X, 
                   nexttree_mask[...,None], noise_X)

            # paranoid checks
            #for x in (deriv_tree_X, attachments_X, elem_trees_X2, deriv_trees_X2,
            #       attachments_Y, elem_trees_Y,  tree_noise):
            #    if np.any(np.isnan(x)):
            #        raise Exception("FOUND A NAN")


            #yield (deriv_tree_X, attachments_X, elem_trees_X2, deriv_trees_X2,
            #       attachments_Y, elem_trees_Y,  tree_noise)

    def report(self):
        if len(self.observations) == 0:
            if self.verbose: 
                raise Exception("I can't report; failing loudly")
            elif self.logger: 
                self.logger.warning("Can't report. Failing silently")
            else: 
                return
        obs_time, obs = self.observations[-1]
        epoch, train_loss, ttree_acc5, tattach_acc5, val_loss, vtree_acc5, vattach_acc5 = obs
        if self.logger:
            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("\tLearning Rate: {}".format(self.learning_rate))
            self.logger.info("\tRegularizing Lambda: {}".format(self.reg_lambda))
            self.logger.info("\tTraining Loss: {}".format(train_loss))
            self.logger.info("\tTraining Tree Accuracy: {}".format(ttree_acc5))
            self.logger.info("\tTraining Attach Accuracy: {}".format(tattach_acc5))
            self.logger.info("\tValidation Loss: {}".format(val_loss))
            self.logger.info("\tValidation Tree Accuracy: {}: ".format(vtree_acc5))
            self.logger.info("\tValidation Atttach Accuracy: {}".format(vattach_acc5))
            self.logger.info("\tLoss Ratio (Val/Train): {}".format(val_loss/train_loss))
        if self.verbose:
            print("Epoch {}".format(epoch))
            print("\tTraining Loss: {}".format(train_loss))
            print("\tValidation Loss: {}".format(val_loss))
            print("\tLoss Ratio (Val/Train): {}".format(val_loss/train_loss))

        self.scribe.record(epoch, type="epoch")
        self.scribe.record(train_loss, type="training loss")
        self.scribe.record(ttree_acc5, type="training tree acc5")
        self.scribe.record(tattach_acc5, type="training attach acc5")
        self.scribe.record(val_loss, type="validation loss")
        self.scribe.record(vtree_acc5, type="validation tree acc5")
        self.scribe.record(vattach_acc5, type="validation attach acc5")

"""

    def get_alternates_deprecated(self, tree_ind, atype):
        alts = self.subsets[atype]
        alt_ids = alts[:,0].nonzero()[0]
        alt_feats = alts[alt_ids]

        trick = np.zeros(alts.shape[0])
        trick[tree_ind] = 1
        target_id = trick[alt_ids].nonzero()[0][0]

        return alt_feats, target_id
"""
