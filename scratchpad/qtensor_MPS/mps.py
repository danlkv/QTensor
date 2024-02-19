import tensornetwork as tn
from mpo import MPO, MPOLayer
import numpy as np


class MPS:
    def __init__(self, tensor_name, N, physical_dim=2) -> None:
        """
        Given: Bond dimension = 1

               (0) physical dim
                       |
        (1) bond dim --@-- (2) bond dim
        """
        self._N = N
        self._physical_dim = physical_dim
        self.name = tensor_name

        if N < 1:
            raise ValueError("Number of tensors should be >= 2")
        # Initialise as |0> = [1.0 0.0
        #                      0.0 0.0]
        if N == 1:
            self._nodes = [
                tn.Node(
                    np.array([1.0, 0.0], dtype=np.complex64),
                    name=tensor_name + str(0),
                )
            ]
        else:
            nodes = [
                tn.Node(
                    np.array(
                        [[1.0], *[[0.0]] * (physical_dim - 1)], dtype=np.complex64
                    ),
                    name=tensor_name + str(0),
                )
            ]
            for i in range(N - 2):
                node = tn.Node(
                    np.array(
                        [[[1.0]], *[[[0.0]]] * (physical_dim - 1)], dtype=np.complex64
                    ),
                    name=tensor_name + str(i + 1),
                )
                nodes.append(node)
            nodes.append(
                tn.Node(
                    np.array(
                        [[1.0], *[[0.0]] * (physical_dim - 1)], dtype=np.complex64
                    ),
                    name=tensor_name + str(N - 1),
                )
            )

            for i in range(1, N - 2):
                tn.connect(nodes[i].get_edge(2), nodes[i + 1].get_edge(1))

            if N < 3:
                tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
            else:
                tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
                tn.connect(nodes[-1].get_edge(1), nodes[-2].get_edge(2))

            self._nodes = nodes

    @staticmethod
    def construct_mps_from_wavefunction(
        wavefunction, tensor_name, N, physical_dim=1
    ) -> "MPS":
        """
        Method to create wavefunction from mps
        """
        if wavefunction.size != physical_dim**N:
            raise ValueError()

        wavefunction = np.reshape(wavefunction, [physical_dim] * N)
        to_split = tn.Node(wavefunction, axis_names=[str(i) for i in range(N)])

        nodes = []
        for i in range(N - 1):
            left_edges = []
            right_edges = []

            for edge in to_split.get_all_dangling():
                if edge.name == str(i):
                    left_edges.append(edge)
                else:
                    right_edges.append(edge)

            if nodes:
                for edge in nodes[-1].get_all_nondangling():
                    if to_split in edge.get_nodes():
                        left_edges.append(edge)

            left, right, _ = tn.split_node(
                to_split, left_edges, right_edges, left_name=tensor_name + str(i)
            )

            nodes.append(left)
            to_split = right
        to_split.name = tensor_name + str(N - 1)
        nodes.append(to_split)

        mps = MPS(tensor_name, N, physical_dim)
        mps._nodes = nodes
        return mps

    @property
    def N(self):
        return self._N

    @property
    def physical_dim(self):
        return self._physical_dim

    def get_mps_nodes(self, original) -> list[tn.Node]:
        if original:
            return self._nodes

        nodes, edges = tn.copy(self._nodes)
        return list(nodes.values())

    def get_mps_node(self, index, original) -> tn.Node:
        return self.get_mps_nodes(original)[index]

    def get_tensors(self, original) -> list[tn.Node]:
        nodes = []

        if original:
            nodes = self._nodes
        else:
            nodes, edges = tn.copy(self._nodes)

        return list([node.tensor for node in nodes])

    def get_tensor(self, index, original) -> tn.Node:
        return self.get_tensors(original)[index]

    def is_unitary():
        pass

    def get_wavefunction(self) -> np.array:
        nodes = self.get_mps_nodes(False)
        curr = nodes.pop(0)

        for node in nodes:
            curr = tn.contract_between(curr, node)

        wavefunction = np.reshape(curr.tensor, newshape=(self._physical_dim**self._N))
        return wavefunction

    def apply_single_qubit_gate(self, gate, index) -> None:
        """
        Method to apply single qubit gate on mps
        Assumption: Gates are unitary

         0
         |
        gate
         |
         1

         |
        MPS

        """
        mps_index_edge = list(self._nodes[index].get_all_dangling())[0]
        gate_edge = gate[1]
        temp_node = tn.connect(mps_index_edge, gate_edge)

        new_node = tn.contract(temp_node, name=self._nodes[index].name)
        self._nodes[index] = new_node

    def apply_two_qubit_gate(self, gate, operating_qubits):
        """
        Method to apply two qubit gates on mps

            0  1
            |  |
            gate
            |  |
            2  3

            a   b
            |   |
             MPS
        """
        # reshape the gate to order-4 tensor
        # Assumimg operating_qubits are sorted
        mps_indexA = (
            self.get_mps_node(operating_qubits[0], True).get_all_dangling().pop()
        )
        mps_indexB = (
            self.get_mps_node(operating_qubits[1], True).get_all_dangling().pop()
        )

        temp_nodesA = tn.connect(mps_indexA, gate.get_edge(2))
        temp_nodesB = tn.connect(mps_indexB, gate.get_edge(3))
        left_gate_edge = gate.get_edge(0)
        right_gate_edge = gate.get_edge(1)

        new_node = tn.contract_between(
            self._nodes[operating_qubits[0]], self._nodes[operating_qubits[1]]
        )
        node_gate_edge = tn.flatten_edges_between(new_node, gate)
        new_node = tn.contract(node_gate_edge)

        left_connected_edge = None
        right_connected_edge = None

        for edge in new_node.get_all_nondangling():
            if self.name in edge.node1.name:
                # Use the "node1" node by default
                index = int(edge.node1.name.split(self.name)[-1])
            else:
                # If "node1" is the new_mps_node, use "node2"
                index = int(edge.node2.name.split(self.name)[-1])

            if index <= operating_qubits[0]:
                left_connected_edge = edge
            else:
                right_connected_edge = edge

        left_edges = []
        right_edges = []

        for edge in (left_gate_edge, left_connected_edge):
            if edge != None:
                left_edges.append(edge)

        for edge in (right_gate_edge, right_connected_edge):
            if edge != None:
                right_edges.append(edge)

        u, s, vdag, _ = tn.split_node_full_svd(
            new_node, left_edges=left_edges, right_edges=right_edges
        )

        new_left = u
        new_right = tn.contract_between(s, vdag)

        new_left.name = self._nodes[operating_qubits[0]].name
        new_right.name = self._nodes[operating_qubits[1]].name

        self._nodes[operating_qubits[0]] = new_left
        self._nodes[operating_qubits[1]] = new_right

    def inner_product(self, wMPS):
        """
        Method to calculate inner product of mps
        """

        T = self.get_mps_nodes(False)
        W = wMPS.get_mps_nodes(False)

        for wNode in W:
            wNode.set_tensor(np.conj(wNode.tensor))

        for i in range(self.N):
            tn.connect(T[i].get_all_dangling().pop(), W[i].get_all_dangling().pop())

        for i in range(self.N - 1):
            TW_i = tn.contract_between(T[i], W[i])
            new_node = tn.contract_between(TW_i, W[i + 1])
            W[i + 1] = new_node

        return np.complex128((tn.contract_between(T[-1], W[-1])).tensor)

    def get_norm(self):
        """
        Method to calculate norm of mps
        """
        val = np.sqrt(self.inner_product(self).real)
        return val

    def left_cannoise(self, i):
        nodes = []
        for i in range(i):
            left_edges = []
            right_edges = []

            for edge in to_split.get_all_dangling():
                if edge.name == str(i):
                    left_edges.append(edge)
                else:
                    right_edges.append(edge)

            if nodes:
                for edge in nodes[-1].get_all_nondangling():
                    if to_split in edge.get_nodes():
                        left_edges.append(edge)

            left, right, _ = tn.split_node(
                to_split, left_edges, right_edges, left_name="q" + str(i)
            )

            nodes.append(left)
            to_split = right
        to_split.name = "q" + str(i)
        nodes.append(to_split)

    def __copy__(self):
        copy_mps = MPS(self.name, self._N, self._physical_dim)
        copy_mps._nodes = self.get_mps_nodes(original=False)
        return copy_mps

    def get_expectation(self, observable, idx):
        mps_copy = self.__copy__()
        mps_copy.apply_single_qubit_gate(observable, idx)
        return mps_copy.inner_product(self)

    def apply_mpo(self, operation: MPO) -> None:
        if operation.is_single_qubit_mpo():
            self.apply_single_qubit_gate(operation._node, operation._indices[0])
        else:
            self.apply_two_qubit_gate(operation._node, *operation._indices)

    def apply_mpo_layer(self, mpo: MPOLayer) -> None:
        # Assume self.N == operation.N
        mpo = mpo.get_mpo_nodes(False)
        mps = self.get_mps_nodes(False)

        for i in range(self.N):
            tn.connect(mpo[i].get_all_dangling()[1], mps[i].get_all_dangling()[0])

        nodes = []
        for i in range(self.N):
            new_node = tn.contract_between(mpo[i], mps[i])
            nodes.append(new_node)

        for i in range(self.N - 1):
            _ = tn.flatten_edges_between(nodes[i], nodes[i + 1])

        self._nodes = nodes
