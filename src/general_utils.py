import numpy as np
from functools import partial, reduce

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]