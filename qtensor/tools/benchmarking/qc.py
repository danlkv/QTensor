"""
Profile the function against many different quantum circuits
"""

import tarfile
import qtree
import io
from pathlib import Path
FILEDIR = Path(__file__).parent / 'circuits'
print(FILEDIR)

def get_bris_circuit(diag=4, layers=24, seed=0):
    """
    Args:
        diag (int): length of diagonal
        layers (int): num of layers, 24 to 40
        seed (int): instance, 0 to 9
    """
    file = FILEDIR / 'bris' / f'bris_{diag}.tar.gz'
    tf = tarfile.open(file)
    exf = tf.extractfile(tf.getmember(f'bris_{diag}_{layers}_{seed}.txt' ) )
    bts = exf.read()
    stream = io.StringIO(bts.decode())
    n, circ = qtree.operators.read_circuit_stream(stream)
    return n, circ

def get_rect_circuit(sidea=4, sideb=None, layers=24, seed=0):
    """
    Args:
        sidea (int): side of rectangle
        sideb (int): defaults to sidea
        layers (int): num of layers, 10 to 80
        seed (int): instance, 0 to 9
    """
    if sideb is None:
        sideb = sidea
    size = f'{sidea}x{sideb}'
    file = FILEDIR / 'rect' / f'{size}.tar.gz'
    tf = tarfile.open(file)
    exf = tf.extractfile(tf.getmember(f'{size}/inst_{size}_{layers}_{seed}.txt' ) )
    bts = exf.read()
    stream = io.StringIO(bts.decode())
    n, circ = qtree.operators.read_circuit_stream(stream)
    return n, circ
