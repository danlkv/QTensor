import pyrofiler
import time

def func1(x,y):
    # use PROF defined in profile_with_context_advanced.py
    @pyrofiler.PROF.cpu(desc='Func 1', reference=(x,y))
    def sleep1():
        time.sleep(.1)
    
    sleep1()
    return 1

def func2(x,y):
    with pyrofiler.PROF.timing(desc='Func 2', reference=(x,y)):
        time.sleep(.2)
    return 1


def original():
    time.sleep(0.3)

def func3(x,y):
    with pyrofiler.PROF.timing(desc='Func 3', reference=(x,y)):
        original()
    return 1

def func45(x,y):

    with pyrofiler.PROF.timing(desc='Func 4', reference = (x,y)):
        time.sleep(0.4)
    
    with pyrofiler.PROF.timing(desc='Func 5', reference = (x,y)):
        time.sleep(0.5)





def total():
    for i in range(2):
        for x in range(3):
            for y in range(3):
                func1(x,y)
                # func2(x,y)
                # func3(x,y)

                # with pyrofiler.PROF.timing("Func 4", reference = (x,y)):
                #     time.sleep(0.4)
