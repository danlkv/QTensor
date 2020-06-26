"""
This module implements API to the PACE 2017
treewidth solvers.
Node numbering starts from 1 and nodes are *consequtive*
integers!
"""
import subprocess
import threading
import sys

ENCODING = sys.getdefaultencoding()


def run_exact_solver(data, command="tw-exact", cwd=None,
                     extra_args=None):
    """
    Runs the exact solver and collects its output

    Parameters
    ----------
    data : str
         Path to the input file
    command : str, optional
         Deafults to "tamaki_tw-exact"
    extra_args : str, optional
         Optional commands to the solver

    Returns
    -------
    output : str
         Process output
    """
    sh = command + " "
    if extra_args is not None:
        sh += extra_args

    process = subprocess.Popen(
        sh.split(), cwd=cwd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    process.stdin.write(data.encode(ENCODING))
    output, error = process.communicate()
    if error:
        raise ValueError(error)

    return output.decode(ENCODING)


def run_heuristic_solver(data, wait_time=1,
                         command="tw-heuristic", cwd=None,
                         extra_args=None):
    """
    Runs the exact solver and collects its output

    Parameters
    ----------
    data : str
         Path to the input file
    wait_time : float
         Waiting time in seconds
    command : str, optional
         Deafults to "./tw-heuristic"
    extra_args : str, optional
         Optional commands to the solver

    Returns
    -------
    output : str
         Process output
    """
    sh = command + " "
    if extra_args is not None:
        sh += extra_args

    process = subprocess.Popen(
        sh.split(), cwd=cwd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)

    def terminate_process():
        process.send_signal(subprocess.signal.SIGTERM)

    timer = threading.Timer(
        wait_time, terminate_process)
    try:
        timer.start()
        process.stdin.write(data.encode(ENCODING))
        output, error = process.communicate()
    finally:
        timer.cancel()

    if error:
        raise ValueError(error)

    return output.decode(ENCODING)


def test_run_heuristic():
    d = run_heuristic_solver('', wait_time=1,
                             command="echo 42")
    assert(int(d) == 42)
