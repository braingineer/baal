from baal.utils.data_structures.singleton import Singleton

class Settings:
    __metaclass__ = Singleton
    def __init__(self, config):
        self.__dict__.update(config)

    @classmethod
    def default(cls):
        config = {"VERBOSE": False,
                  "LIGHT": True}
        return cls(config)
    

    def store(self, key, value):
        self.__dict__[key] = value