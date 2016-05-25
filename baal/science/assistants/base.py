"""
Help out with experiments
"""
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

__all__ = ["Igor"]

class Igor(object):
    def __init__(self, config_file, experiment_description="", exp_suffix=""):
        with open(config_file) as fp:
            config = yaml.load(fp)
            self.__dict__.update(config)
        self.screen_check()
        self.stopwatch = Timer()
        self.observations = []
        self.train_data, self.val_data, self.test_data = None, None, None

        exp_dir = self.experiment_name.replace(" ", "_")+exp_suffix
        self.save_location = "{}/{}".format(self.save_location, exp_dir)

        log_location = "{}/log".format(self.save_location)
        baal.utils.ensure_dir(log_location+'/')
        self.logger = baal.utils.loggers.duallog("igor", file_loc=log_location)


        self.param_location = "{}/params".format(self.save_location)
        baal.utils.ensure_dir(self.param_location+'/')

        self.logger.info("Algorithm Notes: {}".format(experiment_description))
        self.logger.info("Config notes: {}".format(self.trial_notes))

        scribe_location = "{}/scribe".format(self.save_location)
        baal.utils.ensure_dir(scribe_location+'/')
        self.scribe = scribe.Scribe(scribe_location, "experiment_logbook",
                             self.experiment_name)
        self.scribe.record(experiment_description, "description")
        self.scribe.record(self.trial_notes, "notes")
        self.scribe.record(tuple(config.iteritems()), "config")

    def screen_check(self):
        if self.check_before_run:
            print("Are you running this is in screen or tlux?")
            print("1. yes")
            print("2. no but whatever")
            print("3. crap. no. exit please.")
            try:
                s = int(input("Selection [default:2]: "))
                if s==3:
                    print("Exitting now.")
                    raise KeyboardInterrupt
                return
            except:
                return

    def start_experiment(self):
        self.stopwatch.tic()

    def end_experiment(self):
        self.stop_time = self.stopwatch.toc()
        self.stopwatch.freeze()

    def observe(self, item):
        obs = (self.stopwatch.toc(), item)
        self.observations.append(obs)

    def report(self):
        pass

    def load_model(self):
        pass

    def reformat_data(self, formatter):
        self.train_data = formatter(self.train_data)
        self.val_data = formatter(self.val_data)
        self.test_data = formatter(self.test_data)
