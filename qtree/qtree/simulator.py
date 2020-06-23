"""
Test programs to demonstrate various use cases of the
Qtree quantum circuit simulator. Functions in this file
can be used as main functions in the final simulator program
"""
import qtree.operators as ops
import qtree.optimizer as opt
import qtree.graph_model as gm
import qtree.np_framework as npfr
import qtree.utils as utils

from qtree.logger_setup import log


def get_amplitudes_from_cirq(filename, initial_state=0):
    """
    Calculates amplitudes for a circuit in file filename using Cirq
    """
    import cirq
    n_qubits, circuit = ops.read_circuit_file(filename)

    cirq_circuit = cirq.Circuit()

    for layer in circuit:
        cirq_circuit.append(op.to_cirq_1d_circ_op() for op in layer)

    print("Circuit:")
    print(cirq_circuit)
    simulator = cirq.Simulator()

    result = simulator.simulate(cirq_circuit, initial_state=initial_state)
    log.info("Simulation completed\n")

    # Cirq for some reason computes all amplitudes with phase -1j
    return result.final_state


def eval_circuit(n_qubits, circuit, final_state,
                 initial_state=0, measured_final=None,
                 measured_initial=None, pdict={}):
    """
    Evaluate a circuit with specified initial and final states.

    Parameters
    ----------
    n_qubits: int
              Number of qubits in the circuit
    circuit: list of lists
             List of lists of gates
    final_state: int
             Values of measured qubits at the end of the circuit (bra).
             Bitwise coded with the length of
             min(n_qubits, len(measured_final)
    initial_state: int
             Values of the measured qubits at the beginning of the
             circuit (ket). Bitwise coded with the length of
             min(n_qubits, len(measured_initial)
    measured_final: list, default None
             Iterable with the positions of qubits which are measured. If
             not all qubits are measured, then a subset of amplitudes will
             be evaluated
    measured_initial: list, default None
             Iterable with the positions of qubits which are measured
             initially. If not all are measured, then the resulting
             amplitudes will be evaluated for a subset of initial states.
    pdict: dict, default {}
    Returns
    -------
    amplitudes: numpy.array
    """
    # Check which qubits are measured
    all_qubits = set(range(n_qubits))
    if measured_final is None:
        measured_final = tuple(range(n_qubits))
    else:
        if not set(measured_final).issubset(all_qubits):
            raise ValueError(f'measured_final qubits outside allowed'
                             f' range: {measured_final}')

    if measured_initial is None:
        measured_initial = tuple(range(n_qubits))
    else:
        if not set(measured_initial).issubset(all_qubits):
            raise ValueError(f'measured_initial qubits outside allowed'
                             f' range: {measured_initial}')

    # Prepare graphical model
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit, pdict=pdict)

    # Collect free qubit variables
    free_final = sorted(all_qubits - set(measured_final))
    free_bra_vars = [bra_vars[idx] for idx in free_final]
    bra_vars = [bra_vars[idx] for idx in measured_final]

    free_initial = sorted(all_qubits - set(measured_initial))
    free_ket_vars = [ket_vars[idx] for idx in free_initial]
    ket_vars = [ket_vars[idx] for idx in measured_initial]

    if len(free_bra_vars) > 0:
        log.info('Evaluate amplitudes over all final states of qubits:')
        log.info(f'{free_final}')

    if len(free_ket_vars) > 0:
        log.info('Evaluate amplitudes over all initial states of qubits:')
        log.info(f'{free_initial}')

    graph_initial = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    graph = gm.make_clique_on(graph_initial, free_bra_vars+free_ket_vars)

    # Get PEO
    peo_initial, treewidth = gm.get_peo(graph)

    # transform peo so free_bra_vars and free_ket_vars are at the end
    # this fixes the layout of the tensor
    peo = gm.get_equivalent_peo(graph, peo_initial,
                                free_bra_vars+free_ket_vars)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    perm_buckets, perm_dict = opt.reorder_buckets(
        buckets, bra_vars + ket_vars + peo)
    perm_graph, _ = gm.relabel_graph_nodes(
        graph, perm_dict)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # make proper slice dictionaries. We choose ket = |0>,
    # bra = |0> on fixed entries
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)
    slice_dict.update(utils.slice_from_bits(final_state, bra_vars))
    slice_dict.update({var: slice(None) for var in free_bra_vars})
    slice_dict.update({var: slice(None) for var in free_ket_vars})

    # Finally make numpy buckets and calculate
    sliced_buckets = npfr.get_sliced_np_buckets(
        perm_buckets, data_dict, slice_dict)
    result = opt.bucket_elimination(
        sliced_buckets, npfr.process_bucket_np,
        n_var_nosum=len(free_bra_vars+free_ket_vars))

    return result.data


def test_parametric_gates():
    """
    Tests circuit evaluation
    """
    a = ops.placeholder()
    c = [[ops.X(0), ops.Y(1), ops.XPhase(2, alpha=a),
          ops.ZPhase(3, alpha=1)]]
    amps = eval_circuit(4, c, final_state=0, initial_state=1,
                        pdict={a: 2})


def test_eval_circuit(filename='inst_2x2_7_0.txt'):
    import numpy as np

    initial_state = 0
    final_state = 0
    measured_initial = [1, 2, 3]
    measured_final = [0, 1, 2, 3]

    # prepare reference
    # calculate proper slices
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    free_bra_vars = [bra_vars[idx] for idx in range(n_qubits)
                     if idx not in measured_final]
    fixed_bra_vars = [bra_vars[idx] for idx in range(n_qubits)
                      if idx in measured_final]
    free_ket_vars = [ket_vars[idx] for idx in range(n_qubits)
                     if idx not in measured_initial]
    fixed_ket_vars = [ket_vars[idx] for idx in range(n_qubits)
                      if idx in measured_initial]

    slice_dict = utils.slice_from_bits(final_state, fixed_bra_vars)
    slice_dict.update({var: slice(None) for var in free_bra_vars})
    slice_dict.update(
        utils.slice_from_bits(initial_state, fixed_ket_vars))
    slice_dict.update({var: slice(None) for var in free_ket_vars})

    # sort slice in the big endian order for Cirq
    bra_subtensor = tuple([slice_dict[var] for var in bra_vars])
    ket_subtensor = tuple([slice_dict[var] for var in ket_vars])

    reference = np.empty([2]*n_qubits+[2]*n_qubits, dtype=np.complex64)[
        bra_subtensor + ket_subtensor]

    temp_slice_dict = utils.slice_from_bits(initial_state, fixed_ket_vars)
    for ket_state in range(2**len(free_ket_vars)):
        amplitudes = get_amplitudes_from_cirq(filename, ket_state)
        slice_of_amplitudes = amplitudes.reshape(
            [2]*n_qubits)[bra_subtensor]
        temp_slice_dict.update(
            utils.slice_from_bits(ket_state, free_ket_vars))
        partial_ket_subtensor = tuple(
            [temp_slice_dict[var] for var in ket_vars])

        final_shape = slice_of_amplitudes.shape + (1,) * n_qubits
        reference[
            bra_subtensor+partial_ket_subtensor
        ] = slice_of_amplitudes.reshape(final_shape)

    # get the result
    result = eval_circuit(n_qubits, circuit, final_state=final_state,
                          initial_state=initial_state,
                          measured_final=measured_final,
                          measured_initial=measured_initial)
    reference = reference.flatten()
    result = result.flatten()

    # difference
    print('Result:')
    print(np.round(result, 3))
    print('Reference:')
    print(np.round(reference, 3))
    print('Max difference:')
    print(np.max(np.abs(result - reference)))


if __name__ == "__main__":
    test_eval_circuit()
