"""
Here we put all system-dependent constants
"""
import numpy as np
import os
import shutil
from .logger_setup import log

QTREE_PATH = os.path.dirname((os.path.abspath(__file__)))
THIRDPARTY_PATH = os.path.join(QTREE_PATH, 'thirdparty')

# Check for QuickBB
QUICKBB_COMMAND = shutil.which('run_quickbb_64.sh')
if not QUICKBB_COMMAND:
    QUICKBB_COMMAND = shutil.which('quickbb_64')
if not QUICKBB_COMMAND:
    quickbb_path = os.path.join(
        THIRDPARTY_PATH, 'quickbb', 'run_quickbb_64.sh')
    if os.path.isfile(quickbb_path):
        QUICKBB_COMMAND = quickbb_path
if not QUICKBB_COMMAND:
    log.warn('QuickBB solver is unavailable')
        
# Check for Tamaki solver
try:
    TAMAKI_SOLVER_PATH = os.path.dirname(
        shutil.which('tw-exact'))
    if not TAMAKI_SOLVER_PATH:
        tamaki_solver_path = os.path.join(
            THIRDPARTY_PATH, 'tamaki_treewidth')
        if os.path.isdir(tamaki_solver_path):
            TAMAKI_SOLVER_PATH = tamaki_solver_path
except Exception:
    log.warn('Tamaki solver is unavailable')

MAXIMAL_MEMORY = 1e22   # 100000000 64bit complex numbers
NP_ARRAY_TYPE = np.complex64

try:
    import tensorflow as tf
    try:
        TF_ARRAY_TYPE = tf.complex64
    except AttributeError:
        pass
except ImportError:
    pass

