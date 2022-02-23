from pyrofiler import Profiler
import pyrofiler
import child
import numpy as np
from pyrofiler import callbacks
from collections import defaultdict
import time

class MyProfiler(Profiler):
    def __init__(self, callback=None):
        super().__init__(callback=callback)
        self.use_append()

    def get_stats(self, label):
        data = [x['value'] for x in self.data[label]]
        return dict(
            mean=np.mean(data),
            max = np.max(data),
            std = np.std(data),
            min = np.min(data)
        )
    
    # Transform the table to be
    # ref fun1 fun2
    def get_refs(self):
        pass

prof = MyProfiler()

'''
1. Pass the profiler globally
'''
pyrofiler.PROF = prof

'''
2. Wrapping the calback function so that the data is stored in this format:
    data[desc] = dict
'''
default_cb = prof.get_callback() #returns self._callbeck, the default callback in this case
def my_callback(value, desc, reference=0):
    default_cb(dict(reference=reference, value=value), desc) #wrapping, data[desc] = dict()
prof.set_callback(my_callback)





# def my_function(i):
#     with prof.timing('My_func_time', reference=i):
#         time.sleep(i)

def main():

    '''
    Directly calls the function that is wrapped
    '''
    # callbacks.disable_printing()
    child.total()

main()
# print('Pyrofiler data', prof.data)
# print('Pyrofiler main', prof.data['from another file'])


def reorderData(data):
    
    result = defaultdict(dict)
    
    for function in data:
        func_dict = defaultdict(list)
        
        # func_dict = ref -> list(value)
        for ref_val in data[function]:
            reference = ref_val['reference']
            value = ref_val['value']
            func_dict[reference].append(value)
        
        # result => ref -> {func1:[] func2:[]}
        for ref in func_dict:
            times = func_dict[ref]
            result[ref][function] = times
    
    return result


def describeData(data):
    result = defaultdict(dict)
    for ref in data:
        funcs = data[ref]
        for func in funcs:
            times = funcs[func]
            func_desc = dict(
                mean=np.mean(times),
                max = np.max(times),
                std = np.std(times),
                min = np.min(times)
            )
            result[ref][func] = func_desc
    return result


# reordered = reorderData(prof.data)
# described = describeData(reordered)


print(prof.data)