import numpy as np
import pyrofiler
from collections import defaultdict
from pandas import DataFrame
from functools import wraps

from .measure_name import MeasureName

import time


def argument_repr(arg):
    """ Convert argument to a unique representation
    that will be use to recognize arguments
    """
    return repr(arg)


def for_all_methods(decorator):
    """ Apply decorator to each method of an object """
    def decorate(cls):
        for attr in cls.__dir__(): # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                try:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
                except Exception:
                    # Fires when setting __class__ to a function
                    pass
        return cls
    return decorate


class Profiler:
    """
    A class that wraps each method of a namespace or
    class and gives results
    """
    def __init__(self, monitor='time'):
        self.monitor = monitor
        self._measures = defaultdict(list)
        self._class_tracks = {}

    def get_func_df(self, fname):
        df = DataFrame.from_records(self._measures[fname])
        return df

    def get_module_df(self):
        records = [{'fname':name, 'time':self._measures[name][MeasureName('time')]}
                    for name in self._measures
                  ]
        df = DataFrame.from_records(records)
        return df

    def set_class_track(self, cls, func):
        self._class_tracks[cls] = func

    def get_props_from_arg(self, arg):
        return self._class_tracks.get(type(arg), argument_repr)(arg)

    def get_props_from_args(self, *args, **kwargs):
        """ Measured properties over arguments
        Basically, just maps ``get_props_from_arg`` over args and kwargs
        """
        args = [self.get_props_from_arg(a) for a in args]
        kwargs = {k:self.get_props_from_arg(v) for k, v in kwargs.items()}
        return args, kwargs

    def call_function(self, f, *args, **kwargs):
        """ Calls a function and saves time and argument duration. """
        props, kwprops = self.get_props_from_args(*args, **kwargs)
        fname = f.__name__
        start = time.time()

        ret = f(*args, **kwargs)
        end = time.time()
        duration = end - start
        measures = kwprops
        for i, v in enumerate(props):
            measures[i] = v
        measures[MeasureName('time')] = duration
        self._measures[fname].append(measures)
        return ret

    def add(self, function):
        """ Decorator to use for profiling functions

        Example:
            >> from profiler import Profiler
            >> prof = Profiler()
            >> @prof.add
                def foo():
                    print('Hi')
            >> foo()
        """
        # return if somebody actually called the wrapper
        if function is None:
            return self.add

        @wraps(function)
        def wrap(*args, **kwargs):
            return self.call_function(function, *args, **kwargs)
        return wrap

    def get_times(self, fname):
        return [x[MeasureName('time')] for x in self._measures[fname]]

    def add_class(self, cls):
        """ Decorate all methods of a class isinstance with profiler """
        dec = for_all_methods(self.add)
        return dec(cls)
