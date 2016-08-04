from time import time
import collections
from math import floor
import types
import os
#from markdown2 import Markdown


class SimpleProgress:
    """
    A progress function that does linear extrapolation between
      time spent and number of things completed.

    Usage:
      prog = SimpleProgress(number_of_units)
      prog.start()
      for i in range(num_iterations):
        prog.update(i)
        is this working
    """

    def __init__(self, total, web=None):
        self.total = total
        self.count = 0
        self.num_updates = 1
        self.web = web

    def start_progress(self):
        self.start_time = time()

    def start(self):
        self.start_progress()

    def incr(self, amount=1):
        self.count+=amount

    def should_output(self, trigger_percent=0.1):
        if self.count > self.total * trigger_percent * self.num_updates:
            self.num_updates += 1
            return True
        return False

    def should_update(self, trigger_percent=0.1):
        return self.should_output(trigger_percent)

    def output(self, x=None):
        return self.update(x)

    def update(self, x = None):
        if x is None:
            x = self.count

        update_val = ""

        if x > 0:
            elapsed = time() - self.start_time
            percDone = x * 100.0 / self.total
            estimatedTimeInSec = (elapsed / float(x)) * self.total
            update_val = """
                  %s %s percent
                  %s Processed
                  Elapsed time: %s
                  Estimated time: %s
                  --------""" % (self.bar(percDone),
                                 round(percDone, 2),
                                 x, self.form(elapsed),
                                 self.form(estimatedTimeInSec))
        if self.web is not None:
            self.web_update(update_val)
        return update_val

    def web_update(self, update_val):
        with open(self.web+".html", 'a') as fp:
            fp.write("\n\n{}".format(update_val))
        #with open(self.web+".md") as fp:
        #    outhtml = Markdown().convert("".join(fp.readlines()))
        #with open(self.web+".html", 'w') as fp:
        #   fp.write("<html><body>"+outhtml+"</body></html")

    def expiring(self):
        elapsed = time() - self.start_time
        return elapsed / (60.0 ** 2) > 71.

    def form(self, t):
        hour = int(t / (60.0 * 60.0))
        minute = int(t / 60.0 - hour * 60)
        sec = int(t - minute * 60 - hour * 3600)
        return "%s Hours, %s Minutes, %s Seconds" % (hour, minute, sec)

    def bar(self, perc):
        done = int(round(30 * (perc / 100.0)))
        left = 30 - done
        return "[%s%s]" % ('|' * done, ':' * left)

class WelfordStats:
    """
    See these for information:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    http://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
    """
    def __init__(self):
        self.n = 0.0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / n
        self.M2 += delta * (x - mean)

    @property
    def variance(self):
        if self.n <= 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)



def group_by(iterable, func):
    res = defaultdict(lambda: [])
    for item in iterable:
        key = func(key)
        res[key].append(item)
    return res

def process_file(filename):
    fp = open(filename)
    data = [x.replace("\n", "") for x in fp.readlines()]
    fp.close()
    return data


def ensure_file(some_file):
    return os.path.exists(some_file)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        return True
    return False

def bigram_enum(sequence):
    for i, x in enumerate(sequence[1:], 1):
        yield i, sequence[i - 1], x

def backward_enumeration(thelist):
   for index in reversed(xrange(len(thelist))):
      yield index, thelist[index]

def reversezip(xys):
    return [[x[i] for x in xys] for i in range(len(xys[0]))]


def lrange(x, start=0, step=1):
    return range(start, len(x), step)


def get_ind(x, i):
    return [y[i] for y in x]


def reverse_dict(in_dict):
    out_dict = {n: k for k, ns in in_dict.items() for n in ns}
    return out_dict


def empty(x):
    if isinstance(x, collections.Iterable):
        return not len(x) > 0
    raise TypeError


def flatten(some_list):
    """
    assumes list of lists
    for sublist in some_list:
        for item in sublist:
            yield item

    """
    return [item for sublist in some_list for item in sublist]


def count_time(t, d=2):
    seconds = t % 60
    minutes = int(floor(t / 60))
    if d == 2:
        return "{0:02} Min; {1:2.2f} Sec.".format(minutes, seconds)
    raise NotImplementedError


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    index = {'0':HEADER,
             '1':OKBLUE,
             '2':OKGREEN,
             "w":WARNING,
             "f":FAIL,
             "b":BOLD,
             "u":UNDERLINE}


def cprint(x,levels=[0]):
    """ Deprecated for colorama """
    print("".join([bcolors.index[str(level)]
                   for level in levels]) + x + bcolors.ENDC)


def cformat(x, levels=[0]):
    return "".join([bcolors.index[str(level)]
                   for level in levels]) + x + bcolors.ENDC

class easyc:
    @staticmethod
    def red(astr):
        return cformat(astr, ['f'])

    @staticmethod
    def blue(astr):
        return cformat(astr, ['1'])

    @staticmethod
    def green(astr):
        return cformat(astr, ['2'])

    @staticmethod
    def yellow(astr):
        return cformat(astr, ['w'])



def cprint_showcase():
    print(bcolors.BOLD + bcolors.UNDERLINE + "Options Showcase" + bcolors.ENDC)
    for name in bcolors.index.keys():
        cprint("Argument option: %s.  Effect shown." % name, [name])

def nonstr_join(arr,sep):
    if not isinstance(arr,types.ListType):
        arr = [arr]
    return str(sep).join([str(x) for x in arr])

class while_loop_manager(object):
    def __init__(self, condition, log_on=False):
        self.condition = condition
        self.iter_i = 0
        self.log_on = log_on

    def loop(self,loop_indicator):
        while self.condition(loop_indicator):
            if self.log_on:
                self.log()
            self.iter_i += 1
            yield True
        yield False

    def log(self):
        print("Iteration %s: %d" % self.iter_i)
