import matplotlib.pyplot as plt
import theano
from lasagne.layers import Layer, get_output
from lasagne.utils import create_param
from lasagne.init import Constant

__all__ = ["Chef"]

class Chef(object):
    def __init__(self, igor=None):
        self.report_func = None
        self.parameters = {}
        ### trust me, you want an igor
        self.igor = igor


    #########################
    ### parameter management
    ########################

    def make_W(self, num_in, num_out, name):
        P = create_param(self.igor.default_initializer, (num_in, num_out), name=name)
        self.parameters[name] = P
        return P

    def make_b(self, size, name):
        P = create_param(Constant(0.), (size, ), name=name)
        self.parameters[name] = P
        return P

    ######################
    ### layer management
    ######################

    @property
    def layer_iter(self):
        for k,v in self.__dict__.items():
            if isinstance(v, Layer):
                yield (k,v)

    def prep(self, deterministic=False):
        layer_pairs = list(self.layer_iter)
        layers = [v for k,v in layer_pairs]
        names = [k for k,v in layer_pairs]
        outputs = get_output(layers, deterministic=deterministic)
        for name, output in zip(names, outputs):
            out_name = "{}_out".format(name)
            self.__dict__[out_name] = output

    def all_outputs(self):
        out = [v for k,v in self.__dict__.items() if "_out" in k]
        out_names = [k for k,v in self.__dict__.items() if "_out" in k]
        return out, out_names

    def get_layer_outputs(self, names=None, get_all=False):
        if names is None and get_all:
            return all_outputs()
        elif names is not None:
            return [self.__dict__["{}_out".format(n)] for n in names], names
        else:
            raise Exception("bad output request to the chef")   

    #############
    ### reporter
    #############

    def prep_report(self, input_variables):
        self.prep(True)
        out_vals, self.out_names = self.all_outputs()
        self.report_func = theano.function(input_variables, out_vals, 
                                           on_unused_input='ignore',
                                           allow_input_downcast=True)

    def report(self, inputs):
        if self.report_func is None:
            raise Exception("Give the chef some heads up first with `Chef.prep_report`")
        out_vals = self.report_func(*inputs)
        out_dict = dict(zip(self.out_names, out_vals))
        return out_dict


    ##############
    ### grapher
    ##############
    def prep_graph(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2,3)
        self.lines = {}
        self.line_data = {}
        self.batch_index = 0
        self.maxes = {}
        self.mins = {}
        self.refresh_lines()

    def refresh_lines(self, refresh_index=True):
        lines = {}
        line_data = {}
        maxes = {}
        mins = {}
        for i, D in enumerate(['train', 'val']):
            maxes[D] = {}; mins[D] = {}
            lines[D] = {}
            line_data[D] = {}
            for j, L in enumerate(['loss', 'tree accuracy', 'attach accuracy']):
                maxes[D][L] = -1 * (10 ** 5)
                mins[D][L] = 10 ** 5
                self.axes[i,j].set_xlabel(L)
                lines[D][L] = self.axes[i, j].plot([], [])[0]
                line_data[D][L] = {'X':[], "Y":[]}
            if (not refresh_index and 
                D in self.line_data and
                'batch index' in self.line_data[D]):
                line_data[D]['batch index'] = self.line_data[D]['batch index']
            else:
                line_data[D]['batch index'] = 0
        self.lines = lines
        self.line_data = line_data
        if len(self.maxes) == 0:
            self.maxes = maxes
            self.mins = mins
        
    def update_graph(self, xy_dict, D="train"):
        fd_X = lambda d,l: self.line_data[d][l]['X']
        fd_Y = lambda d,l: self.line_data[d][l]['Y']
        f_update = lambda d,l: (self.lines[d][l].set_xdata(fd_X(d,l)), 
                                self.lines[d][l].set_ydata(fd_Y(d,l)))
        b_ind = self.line_data[D]['batch index']
        ax_i = {'train':0, 'val':1}[D]
        for L,new_y in xy_dict.items():
            fd_X(D,L).append(b_ind)
            fd_Y(D,L).append(new_y)
            f_update(D,L)
            self.mins[D][L] = mi = min(min(self.mins[D][L], new_y), 0)
            self.maxes[D][L] = mx = max(max(self.maxes[D][L], new_y), 0)
            ax_j = {'loss':0, 'tree accuracy':1, 'attach accuracy':2}[L]

            self.axes[ax_i, ax_j].set_xlim(0, b_ind)
            self.axes[ax_i, ax_j].set_ylim(mi, mx)

        self.line_data[D]['batch index'] += 1
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def new_epoch(self):
        self.refresh_lines(refresh_index=False)

    def save_graph(self, save_location, alpha, lam):
        plt.savefig("{}/training_graph_{}_{}.png".format(save_location,
                                                         alpha,
                                                         lam))