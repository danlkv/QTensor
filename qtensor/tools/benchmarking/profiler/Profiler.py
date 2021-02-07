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
    return id(arg)


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
    TIME = MeasureName('time')
    def __init__(self, monitor='time'):
        self.monitor = monitor
        self._measures = defaultdict(list)
        self._measure_calcs = defaultdict(list)
        self._class_tracks = {}

    def estimated_overhead_total(self):
        return 7e-5*sum([len(val) for val in self._measures.values()])

    def estimated_overhead_ratio(self):
        """ Returns ratio of overhead to total time """
        return self.estimated_overhead_total()/self.total_time()

    def total_time(self):
        return sum([sum(self.get_times(name)) for name in self._measures.keys()])

    def get_func_df(self, fname):
        df = DataFrame.from_records(self._measures[fname])
        return df

    def get_module_df(self):
        records = [{'fname':name, 'time':sum(self.get_times(name))}
                    for name in self._measures
                  ]
        df = DataFrame.from_records(records)
        return df

    def add_measure_calc(self, fname, measure_name, measure_calc):
        """
        Measure somethig from raw input arguments of a function.

        Args:
            fname (callable | str): function or function name to which measurement applies
            measure_name: name of the measurement variable
            measure_calc: function with same arguments as fname
        """
        if callable(fname):
            fname = fname.__name__
        self._measure_calcs[fname].append((measure_name, measure_calc))

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
        custom_calcs = {calc[0]:calc[1](*args, **kwargs)
                        for calc in self._measure_calcs[fname]
                       }

        ret = f(*args, **kwargs)
        end = time.time()
        duration = end - start
        measures = kwprops
        for i, v in enumerate(props):
            measures[i] = v
        for k in custom_calcs:
            measures[k] = custom_calcs[k]
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
