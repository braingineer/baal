# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time

class Timer(object):
    """A simple timer.

    Note: if frozen, will no longer toc.
    """
    def __init__(self):
        self.total_time = 0.
        self.ncalls = 0
        self.start_time = 0.
        self.total_time = 0.
        self.average_time = 0.
        self.frozen = False

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        if self.frozen:
            return self.total_time
        self.total_time = time.time() - self.start_time
        self.ncalls += 1
        self.average_time_per_call = self.total_time / self.ncalls
        if average:
            return self.average_time_per_call
        else:
            return self.total_time

    def freeze(self):
        self.frozen = True

class EncodeTimer(Timer):
    def __init__(self):
        super(EncodeTimer, self).__init__()
        self.last_toc = 0.

    def toc(self, with_total=False):
        super(EncodeTimer, self).toc()
        toc_diff = self.total_time - self.last_toc
        self.last_toc = self.total_time
        if with_total:
            return toc_diff, self.total_time
        return toc_diff


class EggTimer(Timer):
    def __init__(self, trigger_time):
        super(EggTimer, self).__init__()
        self.trigger_time = trigger_time

    def toc(self):
        diff = time.time() - self.start_time
        if diff > self.trigger_time:
            ### our egg timer is up!
            return True
        else:
            return False
