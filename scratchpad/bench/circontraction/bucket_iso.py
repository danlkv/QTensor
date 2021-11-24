from qtensor.contraction_backends import get_backend
from functools import lru_cache, reduce
from qtree.optimizer import Var, Tensor
import numpy as np
import pyrofiler
import json
import platform

bucket_signatures = [
    "[E369(v_385,v_386,v_387,v_388,v_389,v_390,v_391,v_392,v_395,v_402,v_404,v_405), E384(v_385,v_386,v_387,v_388,v_389,v_390,v_392,v_393,v_394,v_396,v_397,v_398,v_399,v_400,v_401,v_403,v_404)]",
    "[E406(v_411,v_412,v_413,v_414,v_415,v_416,v_420,v_421,v_424,v_425,v_426,v_427,v_429,v_431), E410(v_411,v_412,v_413,v_414,v_415,v_416,v_417,v_418,v_419,v_420,v_421,v_422,v_423,v_428,v_430,v_431)]",
    "[E49(v_348), E343(v_348,v_349,v_350,v_351,v_352,v_353,v_357,v_358,v_362,v_363,v_364,v_365,v_366,v_368), E344(v_348,v_349,v_350,v_351,v_352,v_354,v_355,v_356,v_357,v_359,v_360,v_361,v_366,v_367,v_368)]"
]

def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)

@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None

def signParser(bucSign:str):
    sign = bucSign[1:-1]
    splitedSign = sign.split(", ")
    tensors = []
    for tenSign in splitedSign:
        tenSign = tenSign[:-1]
        tenName, varSigns = tenSign.split("(")
        varSigns = varSigns.split(',')
        vars = ()
        for varSign in varSigns:
            number = varSign[2:]
            currVar = Var(number)
            vars += (currVar,)

        # [] * len(vars)
        varLength = [2] * len(vars)
        a = np.random.rand(*varLength)
        b = np.random.rand(*varLength)
        c = a + 1j*b
        tensor = Tensor(name = tenName, indices = vars, data = c)
        tensors.append(tensor)
    return tensors




if __name__ == '__main__':
    timing = pyrofiler.timing
    backends = ["cupy", "torch_gpu"]
    rep = 10

    for backend in backends:
        be = get_backend(backend)
        for i, sign in enumerate(bucket_signatures):
            times = []
            for _ in range(rep):
                with timing(callback=lambda x: None) as gen_pb:
                    bucket = signParser(sign)
                    be.process_bucket(bucket)
                times.append(gen_pb.result)
            
            GPU_PROPS = get_gpu_props_json()

            report = {}
            report["backend"] = backend
            report["mean_time"] = mean_mmax(times)
            report["platform"] = dict(name=platform.node(), gpu=GPU_PROPS)
            report["bucket_idx"] = i
            print(json.dumps(report))

            
 
            
