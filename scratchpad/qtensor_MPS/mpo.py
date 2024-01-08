import tensornetwork as tn
import numpy as np
from gates import igate
from typing import List, Union, Text, Optional, Any, Type


class MPOLayer:
    def __init__(self, tensor_name, N, physical_dim) -> None:
        """
        (0) physical dim                 (0) physical dim                (0) physical dim
                |                                |                                 |
                @-- (1) bond dim .. (2)bond dim--@-- (1) bond dim ... (1)bond dim--@
                |                                |                                 |
        (2) physical dim                   (3) physical dim                (2) physical dim
        """
        self.tensor_name = tensor_name
        self.N = N
        self.physical_dim = physical_dim

        if N < 2:
            raise ValueError("Number of tensors should be >= 2")
        #  [1.0 0.0
        #   0.0 0.1]
        nodes = [
            tn.Node(
                np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex64),
                name=tensor_name + str(0),
            )
        ]

        for i in range(N - 2):
            node = tn.Node(
                np.array([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=np.complex64),
                name=tensor_name + str(i + 1),
            )
            nodes.append(node)

        nodes.append(
            tn.Node(
                np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex64),
                name=tensor_name + str(N - 1),
            )
        )

        if N < 3:
            tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
        else:
            for i in range(1, N - 2):
                tn.connect(nodes[i].get_edge(1), nodes[i + 1].get_edge(2))
            tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(2))
            tn.connect(nodes[-1].get_edge(1), nodes[-2].get_edge(1))

        self._nodes = nodes

    def get_mpo_nodes(self, original) -> list[tn.Node]:
        if original:
            return self._nodes

        nodes, edges = tn.copy(self._nodes)
        return list(nodes.values())

    def get_mpo_node(self, index, original) -> list[tn.Node]:
        return self.get_mpo_nodes(original)[index]

    def construct_mpo(self, gate_function) -> "MPO":
        # IIZ
        gate_function = gate_function.reshape([self.physical_dim] * 2 * self.N)
        to_split = tn.Node(
            gate_function,
            axis_names=["u" + str(i) for i in range(self.N)]
            + ["d" + str(i) for i in range(self.N)],
        )

        nodes = []

        for i in range(self.N - 1):
            left_edges = []
            right_edges = []

            for edge in to_split.get_all_dangling():
                if edge.name == "u" + str(i) or edge.name == "d" + str(i):
                    left_edges.append(edge)
                else:
                    right_edges.append(edge)

            if nodes:
                for edge in nodes[-1].get_all_nondangling():
                    if to_split in edge.get_nodes():
                        left_edges.append(edge)

            left, right, _ = tn.split_node(
                to_split,
                left_edges,
                right_edges,
                left_name=self.tensor_name + str(i),
                max_singular_values=1,
            )
            nodes.append(left)
            to_split = right
        to_split.name = self.tensor_name + str(self.N - 1)
        nodes.append(to_split)

        self._nodes = nodes

    def add_single_qubit_gate(self, gate_matrix, idx):
        if idx in range(1, self.N):
            gate_matrix = gate_matrix.reshape(
                self.physical_dim, 1, 1, self.physical_dim
            )
        else:
            gate_matrix = gate_matrix.reshape(self.physical_dim, 1, self.physical_dim)

        gate_node = tn.Node(
            gate_matrix,
            name=self.tensor_name + str(idx + 1),
        )

        self._nodes[idx] = gate_node

    def two_qubit_svd(self, new_node, operating_qubits):
        left_connected_edge = None
        right_connected_edge = None

        for edge in new_node.get_all_nondangling():
            if self.tensor_name in edge.node1.name:
                # Use the "node1" node by default
                index = int(edge.node1.name.split(self.tensor_name)[-1])
            else:
                # If "node1" is the new_mps_node, use "node2"
                index = int(edge.node2.name.split(self.tensor_name)[-1])

            if index <= operating_qubits[0]:
                left_connected_edge = edge
            else:
                right_connected_edge = edge

        left_edges = []
        right_edges = []

        # for edge in (left_gate_edge, left_connected_edge):
        #     if edge != None:
        #         left_edges.append(edge)

        # for edge in (right_gate_edge, right_connected_edge):
        #     if edge != None:
        #         right_edges.append(edge)

        left_edges.append(new_node.get_all_dangling()[0])
        left_edges.append(new_node.get_all_dangling()[2])
        left_edges.append(left_connected_edge)

        right_edges.append(new_node.get_all_dangling()[1])
        right_edges.append(new_node.get_all_dangling()[3])
        right_edges.append(right_connected_edge)

        u, s, vdag, _ = tn.split_node_full_svd(
            new_node, left_edges=left_edges, right_edges=right_edges
        )

        new_left = u
        new_right = tn.contract_between(s, vdag)

        new_left.name = self._nodes[operating_qubits[0]].name
        new_right.name = self._nodes[operating_qubits[1]].name

        self._nodes[operating_qubits[0]] = new_left
        self._nodes[operating_qubits[1]] = new_right

    def add_two_qubit_gate(self, gate, operating_qubits):
        """
        Method to apply two qubit gates on mpo

            0  1
            |  |
            gate
            |  |
            2  3

            a   b
            |   |
             MPO
            |   |
            c   d

            0  1
            |  |
            gate'
            |  |
            2  3
        """
        # transpose
        gateT = tn.Node(np.conj(gate.tensor))

        mpo_indexA = self.get_mpo_node(operating_qubits[0], True).get_all_dangling()[0]
        mpo_indexB = self.get_mpo_node(operating_qubits[1], True).get_all_dangling()[0]
        mpo_indexC = self.get_mpo_node(operating_qubits[0], True).get_all_dangling()[1]
        mpo_indexD = self.get_mpo_node(operating_qubits[1], True).get_all_dangling()[1]

        temp_nodesA = tn.connect(mpo_indexA, gate.get_edge(2))
        temp_nodesB = tn.connect(mpo_indexB, gate.get_edge(3))

        left_gate_edge = gate.get_edge(0)
        right_gate_edge = gate.get_edge(1)
        # left_gate_edgeT = gateT.get_edge(2)
        # right_gate_edgeT = gateT.get_edge(3)

        new_node = tn.contract_between(
            self._nodes[operating_qubits[0]], self._nodes[operating_qubits[1]]
        )
        node_gate_edge = tn.flatten_edges_between(new_node, gate)
        new_node = tn.contract(node_gate_edge)

        self.two_qubit_svd(new_node, operating_qubits)

        # Try connecting edges of gate 2, 3
        # Check transposition of gate, try multiindexing gate
        temp_nodesC = tn.connect(mpo_indexC, gateT.get_edge(0))
        temp_nodesD = tn.connect(mpo_indexD, gateT.get_edge(1))

        new_node = tn.contract_between(
            self._nodes[operating_qubits[0]], self._nodes[operating_qubits[1]]
        )
        node_gate_edge = tn.flatten_edges_between(new_node, gateT)
        new_node = tn.contract(node_gate_edge)

        self.two_qubit_svd(new_node, operating_qubits)

    def mpo_mps_inner_prod(self, mps):
        mpo = self.get_mpo_nodes(False)

        mps_original = mps.__copy__()
        mps_original = mps_original.get_mps_nodes(False)

        mps = mps.get_mps_nodes(False)

        for wNode in mps:
            wNode.set_tensor(np.conj(wNode.tensor))

        for i in range(self.N):
            tn.connect(mpo[i].get_all_dangling()[0], mps[i].get_all_dangling()[0])
            tn.connect(
                mpo[i].get_all_dangling()[0], mps_original[i].get_all_dangling()[0]
            )

        for i in range(self.N - 1):
            TW_i = tn.contract_between(mpo[i], mps[i])
            TW_i = tn.contract_between(TW_i, mps_original[i])

            new_node = tn.contract_between(TW_i, mpo[i + 1])
            mpo[i + 1] = new_node

        end_node = tn.contract_between(
            tn.contract_between(mpo[-1], mps[-1]), mps_original[-1]
        )

        inner_prod = np.complex128(end_node.tensor)
        return inner_prod


class MPO:
    def __init__(self, node: tn.Node, idx, physical_dim=2) -> None:
        self._node = node
        self._indices = idx
        self._physical_dim = physical_dim

    def get_node(self, original=True) -> tn.Node:
        if original:
            return self._node
        node_dict, _ = tn.copy([self._node])
        return node_dict[self._node]

    def is_single_qubit_mpo(self) -> True:
        return len(self._indices) == 1

    def is_two_qubit_mpo(self) -> True:
        return len(self._indices) == 2
