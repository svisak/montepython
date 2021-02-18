import copy

class State():

    def __init__(self, **kwargs):
        self._dict = {}
        for key, value in kwargs.items():
            self.set(key, value)

    def get(self, key):
        return copy.deepcopy(self._dict[key])

    def set(self, key, value):
        self._dict[key] = copy.deepcopy(value)

    def dict(self):
        return self._dict
