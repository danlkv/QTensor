"""
Profile the function against many different quantum circuits
"""

import tarfile
import qtree
import io
from pathlib import Path

FILEDIR = Path(__file__).parent / 'circuits'


def read_tar_content_file(archive, filename):
    """
    Args:
        archive: name of the archive file
        filename: file name within the archive
    Returns:
        StringIO
    """
    tf = tarfile.open(archive)
    exf = tf.extractfile(tf.getmember(filename) )
    bts = exf.read()
    stream = io.StringIO(bts.decode())
    return stream



def get_avail_bris_params(diag):
    file = FILEDIR / 'bris' / f'bris_{diag}.tar.gz'
    tf = tarfile.open(file)
    members = tf.getmembers()
    params = [ {
        'diag':diag,
        'layers':int(info.name.split('_')[2]),
        'seed':int(info.name.split('_')[3][:-4]),
    } for info in members]
    return params


def get_bris_circuit(diag=4, layers=24, seed=0):
    """
    Args:
        diag (int): length of diagonal
        layers (int): num of layers, 24 to 40
        seed (int): instance, 0 to 9
    """
    file = FILEDIR / 'bris' / f'bris_{diag}.tar.gz'
    stream = read_tar_content_file(file, f'bris_{diag}_{layers}_{seed}.txt' )
    n, circ = qtree.operators.read_circuit_stream(stream)
    return n, circ


def get_avail_rect_params(sidea, sideb=None):
    if sideb is None:
        sideb = sidea
    size = f'{sidea}x{sideb}'
    file = FILEDIR / 'rect' / f'{size}.tar.gz'
    tf = tarfile.open(file)
    members = tf.getmembers()
    params = [ {
        'sidea':int(info.name.split('_')[1].split('x')[0]),
        'sideb':int(info.name.split('_')[1].split('x')[1]),
        'layers':int(info.name.split('_')[2]),
        'seed':int(info.name.split('_')[3][:-4]),
    } for info in members]
    return params


def get_rect_stream(sidea=4, sideb=None, layers=24, seed=0):
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
    stream = read_tar_content_file(file, f'{size}/inst_{size}_{layers}_{seed}.txt' )
    return stream


def get_rect_circuit(sidea=4, sideb=None, layers=24, seed=0):
    """
    Args:
        sidea (int): side of rectangle
        sideb (int): defaults to sidea
        layers (int): num of layers, 10 to 80
        seed (int): instance, 0 to 9
    """
    stream = get_rect_stream(sidea=sidea, sideb=sideb, layers=layers, seed=seed)
    n, circ = qtree.operators.read_circuit_stream(stream)
    return n, circ
