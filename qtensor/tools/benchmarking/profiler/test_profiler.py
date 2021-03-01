from qtensor.tools.benchmarking import Profiler

class Foo():
    def bark(self):
        print('bark')

    def bark_bark(self, *args, **kwargs):
        self.bark()
        self.bark()

def test_profiler():

    prof = Profiler()

    f = Foo()
    f = prof.add_class(f)
    f.bark_bark()
    print('Measures', prof._measures)

    f.bark_bark()
    print('Measures', prof._measures)
    bark_time = sum(prof.get_times('bark'))
    bark_bark_time = sum(prof.get_times('bark_bark'))
    overhead = 0.5*(bark_bark_time - bark_time)
    print('overhead', overhead)
    assert overhead > 0

    f.bark_bark('test')
    print(prof._measures)
    assert id('test') == prof._measures['bark_bark'][-1][0]

    f.bark_bark('test', 'test2')
    print(prof._measures)
    assert id('test2') == prof._measures['bark_bark'][-1][1]
    f.bark_bark('test', vol=15)
    print(prof._measures)
    assert 'vol' in prof._measures['bark_bark'][-1]
    assert id(15) == prof._measures['bark_bark'][-1]['vol']

def test_tracks():
    prof = Profiler()

    f = Foo()
    f = prof.add_class(f)
    f.bark_bark()

    prof.set_class_track(int, lambda x:x)
    prof.set_class_track(str, lambda x:x)

    f.bark_bark('test', 'test2')
    print(prof._measures)
    assert 'test2' == prof._measures['bark_bark'][-1][1]
    f.bark_bark('test', vol=15)
    assert 15 == prof._measures['bark_bark'][-1]['vol']

    bark_time = sum(prof.get_times('bark'))
    bark_bark_time = sum(prof.get_times('bark_bark'))
    overhead = 0.5*(bark_bark_time - bark_time)
    overhead_per_call = overhead/len(prof._measures['bark_bark'])
    print('overhead', overhead)
    print('overhead_per_call', overhead_per_call)
    assert overhead > 0
