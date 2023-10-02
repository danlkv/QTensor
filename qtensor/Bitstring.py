import numpy as np

class Bitstring:
    """Bitstring class"""
    def __init__(self, bits: 'list[int]', prob=None, dim=2):
        self._bits = list(bits)
        self._s = ''.join(str(b) for b in self._bits)
        self._prob = prob
        self._dim = dim
    
    def __iter__(self):
        for i in self._bits:
            yield int(i)

    def __repr__(self):
        return f'<{self._s}>'
    
    def __len__(self):
        return len(self._bits)
    
    @classmethod
    def str(cls, s: str, **kwargs):
        return cls([int(_s) for _s in s], **kwargs)
    
    @classmethod
    def int(cls, i: int, width, **kwargs):
        dim = kwargs.get('dim', 2)
        return cls(list(int(_i) for _i in np.unravel_index(i, [dim]*width)), **kwargs)
    
    def __add__(self, other: 'Bitstring'):
        assert self._dim == other._dim
        if self._prob is not None and other._prob is not None:
            return Bitstring(self._bits + other._bits, self._prob * other._prob, dim=self._dim)
        return Bitstring(self._bits + other._bits, dim=self._dim)

    def __iadd__(self, other: 'Bitstring'):
        assert self._dim == other._dim
        if self._prob is not None and other._prob is not None:
            self._prob *= other._prob
        self._bits += other._bits

    def __eq__(self, other: 'Bitstring'):
        return self.to_int() == other.to_int()
    
    def __hash__(self):
        if self._prob is not None:
            return hash((self._s, self._prob, self._dim))
        return int(self.to_int())
    
    def to_int(self):
        return np.ravel_multi_index(self._bits, [self._dim]*len(self._bits))
