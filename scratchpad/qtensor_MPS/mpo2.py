import tensornetwork as tn
import numpy as np
from typing import List, Union, Text, Optional, Any, Type
from mps import MPS


class MPO2:
    def __init__(self, nodes: List, name: Text) -> None:
        temp = []
        N = len(nodes)
        """
        (2) physical dim                 (2) physical dim                (2) physical dim
                |                                |                                 |
        0  ---  @-- (1) bond dim .. (0)bond dim--@-- (1) bond dim ... (0)bond dim--@ --- 1
                |                                |                                 |
        (3) physical dim                   (3) physical dim                (3) physical dim
        """
        for node in nodes:
            temp.append(tn.Node(node))

        for i in range(1, N - 2):
            tn.connect(temp[i].get_edge(1), temp[i + 1].get_edge(0))

        if N < 3:
            tn.connect(temp[0].get_edge(1), temp[1].get_edge(0))
        else:
            tn.connect(temp[0].get_edge(1), temp[1].get_edge(0))
            tn.connect(temp[-2].get_edge(1), temp[-1].get_edge(0))

        self.tensors = temp
        self.name = name
        self.N = N

    def get_mpo_nodes(self, original) -> list[tn.Node]:
        if original:
            return self.tensors

        nodes, edges = tn.copy(self.tensors)
        return list(nodes.values())

    def get_mpo_node(self, index, original) -> list[tn.Node]:
        return self.get_mpo_nodes(original)[index]

    def mpo_mps_prod(self, mps: MPS):
        mpo = self.get_mpo_nodes(False)

        mps_original = mps.__copy__()
        mps_original = mps_original.get_mps_nodes(False)

        mps = mps.get_mps_nodes(False)

        for wNode in mps:
            wNode.set_tensor(np.conj(wNode.tensor))

        # tn.connect(mpo[0].get_all_dangling()[-1], mps[0].get_all_dangling()[0])
        # tn.connect(mpo[0].get_all_dangling()[-1], mps_original[0].get_all_dangling()[0])

        for i in range(self.N):
            tn.connect(mpo[i].get_all_dangling()[-1], mps[i].get_all_dangling()[0])
            tn.connect(
                mpo[i].get_all_dangling()[-1], mps_original[i].get_all_dangling()[0]
            )
            # tn.connect(
            #     mpo[i].get_all_dangling()[0], mps_original[i].get_all_dangling()[0]
            # )

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


class FiniteTFI:
    """
    The famous transverse field Ising Hamiltonian.
    The ground state energy of the infinite system at criticality is -4/pi.

    Convention: sigma_z=diag([-1,1])
    """

    def __init__(self, Jx: np.ndarray, Bz: np.ndarray, name: Text = "TFI_MPO") -> None:
        """
        Returns the MPO of the finite TFI model.
        Args:
          Jx:  The Sx*Sx coupling strength between nearest neighbor lattice sites.
          Bz:  Magnetic field on each lattice site.
          dtype: The dtype of the MPO.
          backend: An optional backend.
          name: A name for the MPO.
        Returns:
          FiniteTFI: The mpo of the infinite TFI model.
        """
        dtype = np.complex64
        N = len(Bz)
        sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
        sigma_z = np.diag([-1, 1]).astype(dtype)
        mpo = []
        temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
        # Bsigma_z
        temp[0, 0, :, :] = Bz[0] * sigma_z
        # sigma_x
        temp[0, 1, :, :] = Jx[0] * sigma_x
        # 11
        temp[0, 2, 0, 0] = 1.0
        temp[0, 2, 1, 1] = 1.0
        mpo.append(temp)
        for n in range(1, N - 1):
            temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
            # 11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0
            # sigma_x
            temp[1, 0, :, :] = sigma_x
            # Bsigma_z
            temp[2, 0, :, :] = Bz[n] * sigma_z
            # sigma_x
            temp[2, 1, :, :] = Jx[n] * sigma_x
            # 11
            temp[2, 2, 0, 0] = 1.0
            temp[2, 2, 1, 1] = 1.0
            mpo.append(temp)

        temp = np.zeros([3, 1, 2, 2], dtype=dtype)
        # 11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        # sigma_x
        temp[1, 0, :, :] = sigma_x
        # Bsigma_z
        temp[2, 0, :, :] = Bz[-1] * sigma_z
        mpo.append(temp)

        self.tensors = mpo
        self.name = name


N = 4
Jz = np.ones(4)
Bz = np.ones(4)
tfi = FiniteTFI(Jz, Bz)
mpo = MPO2(tfi.tensors, tfi.name)
mps1 = MPS("q", N, 2)
print(mpo.mpo_mps_prod(mps1))
