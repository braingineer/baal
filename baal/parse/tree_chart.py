"""
Tree Chart for Parsing a Tree Grammar

A parser which uses chart edges in an ordered agenda to find suitable parses
for strings using a tree grammar

@author bcmcmahan

Description:
    Advanced by Martin Kay and his colleagues (Kalpan, 1973; Kay, 1982)
    The key principle is the Fundamental Rule (FR)
    The FR states that when a chart contains two contiguous edges, where
        one of the edges provides the constituent that the other needs,
        a new edge should be created that spans the original edges and
        incorporates the new material.

Data Structures:
    1. Agenda
        - Determines how to order the edges.
        - CYK seems to order them based on lexical span length
        - Earley seems to order based on last span index (first to last)
        - Though, I'm not certain on these special chart cases
    2. Edge
        - The main workhourse data structure
        - will manage the entries and the edge index
        - entries are wrappers around trees which maintain extra book keeping
        - the left and rights are based on the span trees in the entry
                (for example, an entry may be made of 2 trees: one subbed into
                    the other)

Algorithm:
    Function chart_parse
    Input: words, grammar, agenda-strategy
    1. agenda, chart = initialize(words, grammar, agenda_strategy)
    2. While agenda is not empty:
        1. current_edge = agenda.pop()
        2. process_edge(current_edge)
    Output: chart

    Function initialize
    Input: words, grammar, adenda_strategy
    # words are assumed to be the input string
    1. agenda = ChartAgenda(agenda_strategy) # maybe similar to Vocabulary class
    2. chart = Chart() # in practice, these will probably all be in chart obj
    3. for w_i, word in enumerate(words):
        # look up all trees with a lexical node of word
        agenda.add_many(grammar.from_word(word), w_i)
    Output: agenda, chart

    Function process_edge
    Input: edge
    1. add_to_chart(edge)
    2. If edge is incomplete:
        1. forward_fundamental_rule(edge) # ins/sub trees INTO edge
    3. Else:
        1. backward_fundamental_rule(edge) # ins/sub edge INTO trees
    Output: None

    Function forward_fundamental_rule
    Input: edge  # not necessarily incomplete. could be looking for insertions
    for complete_edge in from_frontier_buckets(edge, True): # bucket defined later
        for new_edge in edge.combine(complete_edge): # complete edge is an insert or subst
            add_to_agenda(new_edge)
    Output: None

    Function backward_fundamental_rule
    Input: complete_edge # edge is complete
    for edge in from_frontier_buckets(complete_edge, False):
        for new_edge in edge.combine(complete_edge):
            add_to_agenda(new_edge)
    Output: None

    Function add_to_chart
    Input: edge
    1. if hash(edge) not in chart:
        1. chart.add(edge)
    Output:

    Function add_to_agenda
    Input: edge
    1. if hash(edge) not in agenda or agenda_graveyard:
        1. agenda.include(edge)  # should use the agenda strategy
    Output:

    Function from_frontier_buckets
    Input: edge, complete_only
    1. edge_left,edge_right = edge.get_boundaries
    2. for possible_edge in chart.right_buckets[edge_left] + \
                            chart.left_buckets[edge_right]:
        1. if possible_edge.complete and complete_only:
            yield possible_edge
        2. elif not complete_only:
            yield possible_edge
    Output: possible_edge generator
"""
from Queue import PriorityQueue
from nltk.tokenize import wordpunct_tokenize as tokenizer
from baal.nlp.grammars import grammar_api, jurafskymartin_L1, toy_gist
from baal.nlp.semantics import simple_hlf
from baal.utils.general import count_time
from baal.utils import config
from itertools import chain as iterchain
import logging
import sys
import time


# //////////////////////////////////////////////
# Chart Utilities
# //////////////////////////////////////////////


# //////////////////////////////////////////////
# Agendas and Edges
# //////////////////////////////////////////////


class ChartEdge(object):
    def __init__(self, entry, index):
        self.entry = entry
        self.index = index

    @classmethod
    def make(cls, something):
        # don't know what we'd want this for yet.
        pass

    @classmethod
    def make_many(cls, grammar_iterable, index):
        return (cls(entry,index) for entry in grammar_iterable)

    @property
    def boundaries(self):
        return self.index, self.index+len(self.entry.lexical)

    @property
    def complete(self):
        e = self.entry
        return len(e.subst_points) == 0 # and len(e.tree.children) > 0

    def combine(self, other):
        edge_conditionals = (self.boundaries[0] == other.boundaries[1],
                             self.boundaries[1] == other.boundaries[0])
        left_edge,right_edge = edge_conditionals
        for new_entry in self.entry.combine(other.entry, edge_conditionals):
            if left_edge:
                new_index = other.boundaries[0]
            else:
                new_index = self.index
            yield ChartEdge(new_entry, new_index)

    def __hash__(self):
        # roothash = hash(self.entry.tree.symbol)
        # headhash = hash(self.entry.tree.head)
        # subhash = sum([hash(address)+hash(point.symbol)
        #                for address,point in self.entry.subst_points])
        # indexhash = hash(self.boundaries)
        return hash(repr(self.entry.tree))+hash(tuple(self.boundaries))
        # return roothash + headhash + indexhash + subhash

    def __str__(self):
        # f = lambda x: "%s(%s)" % (x.node_type_short, x.symbol)
        # frontier_items = ",".join(f(x) for x in self.tree.frontier.items)
        l,r = self.boundaries
        selfname = "%s" % (self.entry.tree.symbol)
        return "Edge<%s><%s,%s>" % (repr(self.entry.tree), l, r)

    def __len__(self):
        return self.boundaries[1]-self.boundaries[0]

    def __eq__(self, other):
        return hash(other) == hash(self)


class ChartAgenda(object):
    CKY_STRATEGY = "CKY"

    def __init__(self, strategy):
        """ Put strategies into here.
            Add their name as a constant
            Add their function as class function """
        strategy_options = {self.CKY_STRATEGY:ChartAgenda._shortest_str_strategy}
        self.strategy = strategy_options[strategy]
        self.settings = config.ChartSettings.default()
        self.priority_queue = PriorityQueue()
        self._put = self.strategy(self.priority_queue)
        self.graveyard = set()
        self.manifest = set()

    @staticmethod
    def _shortest_str_strategy(priority_queue):
        def put(item):
            priority_queue.put((len(item), item))
        return put

    def _existence_check(self, item):
        gravecheck = item in self.graveyard
        manifestcheck = item in self.manifest
        return manifestcheck or gravecheck

    def _length_check(self, item):
        if self.settings.aggressively_prune:
            if item.boundaries[1] > self.settings.utter_len:
                return False
        return True

    def add_many(self, iterable):
        for item in iterable:
            if not self._existence_check(item):
                self._put(item)
                self.manifest.add(item)

    def add(self, item):
        if not self._existence_check(item) and self._length_check(item):
            self._put(item)
            self.manifest.add(item)

    def pop(self):
        edge_len, edge = self.priority_queue.get()
        self.manifest.remove(edge)
        self.graveyard.add(edge)
        return edge

    def clear(self):
        self.priority_queue = PriorityQueue()
        self._put = self.strategy(self.priority_queue)
        self.graveyard = set()
        self.manifest = set()

    def __len__(self):
        return self.priority_queue.qsize()


# ///////////////////////////////////////////////
# Charts and Such
# ///////////////////////////////////////////////


class TreeChart(object):
    def __init__(self, grammar, strategy=None):
        if not strategy:
            strategy = ChartAgenda.CKY_STRATEGY
        self.settings = config.ChartSettings()
        self.agenda = ChartAgenda(strategy)
        self.grammar = grammar
        self.left_buckets = []
        self.right_buckets = []
        self.logger = logging.getLogger('chart')

    def _on_start(self, utterance):
        # do all on start things
        # maybe clear all chart data structures
        # maybe clear agenda data structures
        self.agenda.clear()
        tokenized_utterance = tokenizer(utterance)
        self.utter_len = self.settings.utter_len = len(tokenized_utterance)
        self.left_buckets = [set() for _ in xrange(self.utter_len+1)]
        self.right_buckets = [set() for _ in xrange(self.utter_len+1)]
        self.initialize_agenda(tokenized_utterance)
        # Buckets are over dot indices, so are len=1
        # self._print_buckets()

    def _print_buckets(self):
        for b_i,bucket in enumerate(self.left_buckets):
            print "\nLeft Bucket #%s" % b_i
            for edge in bucket:
                print edge

        for b_i,bucket in enumerate(self.right_buckets):
            print "\nRight Bucket #%s" % b_i
            for edge in bucket:
                print edge

    def _filter_equivs(self, seen, edge, hlf):
        if hash(edge) in seen[0] or hlf in seen[1]:
            return False
        seen[0].add(hash(edge))
        seen[1].add(hlf)
        return True

        t1_terms = t1.terms
        for terms in seen:
            if terms is None:
                continue
            if t1_terms is None:
                return False
            if t1_terms.keys() != terms.keys():
                continue
            keypass = False
            for key in t1_terms.keys():
                same_var_test = t1_terms[key] == terms[key]
                extended_lf_test = len(t1_terms[key]) == len(terms[key])
                if not same_var_test:
                    keypass = True
                if not extended_lf_test:
                    keypass = True
            if keypass:
                continue
            return False
        return True

    def _on_finish(self, from_terminal):
        if from_terminal: print "finished!"
        seen = (set(), set())
        final = (list(), list())
        filter_sem = self.settings.filter_semantic_equivs
        final_num = 0
        for edge in self.left_buckets[0]:
            hlf = simple_hlf.from_addressbook(edge.entry.addressbook)
            hlf_formatted = simple_hlf.hlf_format(hlf)

            if True:
                is_novel = self._filter_equivs(seen, edge, hlf_formatted)
            else:
                is_novel = True

            spans_chart = edge.boundaries[1] == self.utter_len

            if is_novel and spans_chart:
                final[0].append(edge)
                final[1].append(hlf)
                final_num += 1

                if from_terminal: print "---\n-------------------"
                if from_terminal: print edge.entry
                if from_terminal: edge.entry.print_derived()
                if from_terminal:
                    print "======\nHLF: %s\n======" % hlf_formatted

                if from_terminal: print "-------------------\n---"
        if from_terminal: print "Found %s parses" % final_num
        if from_terminal:
            for edge,hlf in zip(*final):
                hlf_formatted = simple_hlf.hlf_format(hlf)
                print repr(edge.entry.tree), hlf_formatted
        return final

    def initialize_agenda(self, tokenized_utterance):
        for w_i, word in enumerate(tokenized_utterance):
            edges = list(ChartEdge.make_many(self.grammar[word], w_i))
            # print "First index: %s for %s" % (w_i, word)
            self.agenda.add_many(edges)
            self.update_many(edges)
        return len(tokenized_utterance)

    def parse(self, utterance, from_terminal=True):
        start = time.time()
        self._on_start(utterance)

        if from_terminal: print "Agenda is %s long at start" % len(self.agenda)

        while len(self.agenda) > 0:
            edge = self.agenda.pop()
            self.logger.debug("Processing for %s" % edge)
            self.process_edge(edge)

        if from_terminal: print "Completed. Took %s time." % count_time((time.time()-start))
        return self._on_finish(from_terminal)

    def process_edge(self, edge):
        self.update(edge)
        if edge.complete:
            self.backward_fundamental_rule(edge)
        else:
            self.forward_fundamental_rule(edge)

    def forward_fundamental_rule(self, edge):
        self.logger.debug("Forward FR for %s" % edge)
        """ This fundamental rule applies to non complete trees
            It is searching for possible substitution and insertion trees"""
        for complete_edge in self.from_frontier_buckets(edge, True):
            self.logger.debug("Checking for combinations with %s" % edge)
            for new_edge in iterchain(edge.combine(complete_edge),
                                      complete_edge.combine(edge)):
                self.logger.debug("Produced %s" % new_edge.entry)
                self.agenda.add(new_edge)

    def backward_fundamental_rule(self, complete_edge):
        """ This fundamental rule applies to complete trees
            It is searching for non complete trees and insertion trees
            We could probably skip complete, non insertion nodes here
            Decided not to because incoming edge could be an inserter"""
        self.logger.debug("Backward FR for %s" % complete_edge)
        for edge in self.from_frontier_buckets(complete_edge, False):
            self.logger.debug("Checking for combinations with %s" % edge)
            for new_edge in iterchain(edge.combine(complete_edge),
                                      complete_edge.combine(edge)):
                self.logger.debug("Produced %s" % new_edge.entry)
                self.agenda.add(new_edge)

    def update(self, edge):
        """ add it to buckets so that it can be found later """
        edge_left, edge_right = edge.boundaries
        if edge_left >= 0:
            self.left_buckets[edge_left].add(edge)
        if edge_right <= self.utter_len:
            self.right_buckets[edge_right].add(edge)
        return
        try:
            self.left_buckets[edge_left].add(edge)
            self.right_buckets[edge_right].add(edge)
        except IndexError as e:
            raise e

    def update_many(self, edges):
        for edge in edges:
            self.update(edge)

    def from_frontier_buckets(self, edge, complete_only=False):
        edge_left, edge_right = edge.boundaries
        bad_left = edge_right >= len(self.left_buckets)
        bad_right = edge_left < 0
        left_poss = set() if bad_left else self.left_buckets[edge_right]
        right_poss = set() if bad_right else self.right_buckets[edge_left]
        for possible_edge in left_poss.union(right_poss):
            if possible_edge.complete and complete_only:
                yield possible_edge
            elif not complete_only:
                yield possible_edge


def L1_tests():

    grammar = grammar_api.Grammar(jurafskymartin_L1.L1_rules)
    # for tree in grammar_api.make(jurafskymartin_L1.L1_rules):
    #     print tree

    chart = TreeChart(grammar)
    chart.parse("book that flight", True)
    chart.parse("I prefer a book", True)
    chart.parse("I prefer a book to a flight", True)


def simpler_test():
    grammar_rules = """
                    (S (NP) (VP (V loves) (NP)))
                    (NP John)
                    (NP Sandy)
                    (VP (VP*) (ADVP madly))
                    """
    grammar = grammar_api.Grammar(grammar_rules)
    chart = TreeChart(grammar)
    chart.parse("John loves Sandy madly", True)

def harder_tests():
    hal_rules = toy_gist.hal_rules
    data_rules = toy_gist.data_rules
    marvin_rules = toy_gist.marvin_rules


def tail_tests():
    being_safe = False
    blank_grammar = grammar_api.Grammar()
    config.ChartSettings().verbose = True
    chart = TreeChart(blank_grammar)
    next_input = "initial"
    while next_input is not "":
        next_input = raw_input('Type sentence (blank + enter for exit)>')
        if being_safe:
            try:
                chart.parse(next_input)
            except KeyboardInterrupt as E:
                print "Fine.  I'm Leaving.  Geeze"
                break
            except:
                print "%s went wrong.  Check it out. failing gracefully for now" % \
                    (sys.exc_info(),)

        else:
            chart.parse(next_input)


def setup_logger(name,level):
    print name
    level = {"debug": logging.DEBUG, "warning":logging.WARNING}[level]
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    logger.setLevel(level)

if __name__ == "__main__":
    setup_logger('chart','warning')
    setup_logger('trees','warning')
    logging.getLogger('chart').setLevel(logging.DEBUG)
    logging.getLogger('trees').setLevel(logging.DEBUG)
    # L1_tests()
    # simpler_test()
    tail_tests()
