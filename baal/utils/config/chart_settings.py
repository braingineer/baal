from baal.utils.data_structures.singleton import Singleton

class ChartSettings:
    __metaclass__ = Singleton
    def __init__(self,
                 filter_semantic_equivs=False,
                 aggressively_prune=True):

        self.filter_semantic_equivs = filter_semantic_equivs
        self.aggressively_prune = aggressively_prune
        self.utter_len = -1
        self.verbose = False

    @classmethod
    def default(cls):
        return cls()
