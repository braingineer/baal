from __future__ import print_function, division
from sqlitedict import SqliteDict
from datetime import datetime
import itertools
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Scribe", "Transcribe"]

class Scribe(object):
    def __init__(self, location, table_name, exp_name):
        filename = "{}/scribe.sqlite".format(location)
        self.book = SqliteDict(filename, autocommit=True, tablename=table_name)
        unique_id = datetime.now().strftime("date_%m.%d_time_%H.%M")
        self.exp_name = exp_name+"_"+unique_id
        self.observation_index = 0


    def record(self, value, type="general"):
        key = "{}; {}; {}".format(self.exp_name, self.observation_index, type)
        self.book[key] = value
        self.observation_index += 1

    observe = record #sometimes i forget which

    def lookup(self, type=None, exp_name=None, ret_sorted=False, strip_keys=False):
        type_func = lambda *args: True
        name_func = lambda *args: True

        if type:
            type_func = lambda x: x[2] == type

        if exp_name:
            name_func = lambda x: exp_name in x[0]

        key_func = lambda x: type_func(x) and name_func(x)
        unpack = lambda x: [f(x.strip()) for f,x in zip([str,int,str],x.split(";"))]
        items = {k:v for k,v in self.book.iteritems() if key_func(unpack(k))}
        if ret_sorted:
            return self.sort_results(items, strip_keys)
        return items

    def sort_results(self, result_dict, only_val_return=False):
        unpack = lambda x: [f(x.strip()) for f,x in zip([str,int,str],x.split(";"))]
        ranker = lambda x: unpack(x[0])[1]
        sorted_items = sorted(result_dict.items(), key=ranker)
        if only_val_return:
            return [v for k,v in sorted_items]
        return sorted_items

    def close(self):
        self.book.close()


class Transcribe(Scribe):
    def __init__(self, location, table_name='experiment_logbook',
                       find_all=True):
        self.books = {}
        search_string = os.path.join(location, "*", "")
        for trial_folder in glob.glob(search_string):
            try:
                file_loc = os.path.join(trial_folder,
                                        "scribe", "scribe.sqlite")
                new_book = SqliteDict(file_loc, flag='r', tablename=table_name)
                self.books[trial_folder] = new_book
            except Exception as e:
                print(str(e))
                print("failure on {}: {}".format(trial_folder, file_loc))
                continue

    def index_cleaner(self, index_list):
        out = []
        toberemoved = []
        print(index_list)
        for ind_index, ind_val in enumerate(index_list[::-1], 1):
            if ind_val in out:
                toberemoved.append(len(index_list)-ind_index)
            else:
                out.append(ind_val)
        print(out[::-1])
        return out[::-1], toberemoved

    def apply_cleaner(self, alist, inds):
        ret = itertools.ifilter(lambda i_x: i_x[0] not in inds, enumerate(alist))
        ret = [x[1] for x in ret]
        return ret

    def get_config(self):
        config = self.lookup(type='config')
        if len(config) > 1:
            config = max(config.items(), key=lambda (k,v): float(k.split(";")[0][-4:]))
            config = config[1]
        else:
            config = config.values()[0]
        config = dict(config)
        return config

    def grab_data(self, val_keys, index_val, save_path=None):
        all_results = {}
        if len(val_keys) > 1 and index_val in val_keys:
            val_keys.remove(index_val)
        for trial_path, book in self.books.items():
            self.book = book
            all_results[trial_path] = {}

            config = self.get_config()
            all_results[trial_path]['config'] = config

            ### get our x axis.. the epochs usually.
            xvals, removeinds = self.index_cleaner(self.lookup(type=index_val,
                                                               ret_sorted=True,
                                                               strip_keys=True))
            all_results[trial_path][index_val] = xvals

            # get the values we came for
            for val_key in val_keys:
                results = self.lookup(type=val_key, ret_sorted=True, strip_keys=True)
                if len(results) == 0:
                    print("{} not in {}".format(val_key, trial_path))
                    continue
                results = self.apply_cleaner(results, removeinds)
                all_results[trial_path][val_key]=results
            all_results[trial_path]['config']['save_path'] = save_path or trial_path
        return all_results


    def aggregate_plot(self, groups, index_val, all_data, style=None):
        style = style or {'lw':3}

        ###### now plot it
        for trial_path, data in all_data.items():

             ## exit if there's nothing to plot
            if len(data) == 0:
                print("{} has nothing to show.. sad day".format(trial_path))
                continue

            fig, axes = plt.subplots(len(groups), 1, figsize=(10,20))
            ## in case we have only 1 grouping, pyplot results the ax itself
            if len(groups) == 1:
                axes = [axes]

            for ax, group in zip(axes, groups):
                for val in group:
                    ax.plot(data[index_val], data[val], label=val, **style)
                ax.legend()
            ## save it
            save_file = "{}.png".format(data['config']['experiment_name'])
            save_loc = os.path.join(data['config']['save_path'], save_file)
            print("Saving at {}".format(save_loc))

            plt.savefig(save_loc)



if __name__ == "__main__":
    tb = Transcribe("/home/cogniton/research/code/baal/baal/science/gist/pipeline/scripts/output")
    data_keys = ['training loss', 'training acc5', 'training acc8', 'validation acc5', 'validation acc8', 'validation loss']
    save_path = "/home/cogniton/research/code/baal/baal/science/gist/images"
    all_results = tb.grab_data(data_keys,'epoch', save_path=save_path)
    groups = [('training loss', 'validation loss'), ('training acc5', 'training acc8', 'validation acc5', 'validation acc8')]
    tb.aggregate_plot(groups, 'epoch', all_results)
