#from qtensor.noise_simulator import NoiseGate
# from NoiseGate import *
# import attr

class NoiseModel:
    _1qubit_gate = set([
        'M', 'I', 'H', 'Z', 'T', 'Tdag', 'S', 'Sdag', 'X_1_2', 'Y_1_2', 'W_1_2'
        'XPhase', 'YPhase', 'ZPhase', 'U', 

    ])
    _2qubit_gate = set([
        'cX', 'cY', 'cZ', 'SWAP', 'fSim'
    ])
    _3qubit_gate = set([
        'ccX'
    ])

    def __init__(self):
        self.noise_gates = {}
        # not used currently. Just here in case it needs to be referenced for some later use. Channels is accessed in NoiseGate class
        self.channels = []

    # applies noise to all qubits that a gate is acting on
    def add_channel_to_all_qubits(self, channel, gates):
        for gate in gates:
            if gate in self.noise_gates:
                self._check_gate_with_channel(gate, channel)
                channel.add_qubits()
                self.noise_gates[gate].add_channel(channel)
            else: 
                self._check_gate_with_channel(gate, channel)
                channel.add_qubits()
                noise_gate = NoiseGate(gate)
                noise_gate.add_channel(channel)
                self.noise_gates[gate] = noise_gate
                self.channels.append(channel)

    # applies noise to only specific qubits that a gate is acting on. Only matters
    # for multi qubit gates. Eg: if a gate is 'cx' applied to qubits (0, 1), but the
    # list qubits = [1], then noise from channel will only be applied to qubit 1
    def add_channel_to_specific_qubits(self, channel, gates, qubits):
        for gate in gates:
            if gate in self.noise_gates:
                self._check_gate_with_channel(gate, channel)
                channel.add_qubits(qubits)
                self.noise_gates[gate].add_channel(channel)
            else: 
                self._check_gate_with_channel(gate, channel)
                channel.add_qubits(qubits)
                noise_gate = NoiseGate(gate)
                noise_gate.add_channel(channel)
                self.noise_gates[gate] = noise_gate
                self.channels.append(channel)
                
    def _check_gate_with_channel(self, gate, channel):
        if gate in self._1qubit_gate and channel.num_qubits != 1:
            raise Exception("{} qubit channel ".format(channel.num_qubits) + \
                            "cannot be applied to 1-qubit gate {}".format(gate))
        if gate in self._2qubit_gate and channel.num_qubits != 2:
            raise Exception("{} qubit channel ".format(channel.num_qubits) + \
                            "cannot be applied to 2-qubit gate {}".format(gate))
        if gate in self._3qubit_gate and channel.num_qubits != 3:
            raise Exception("{} qubit channel ".format(channel.num_qubits) + \
                            "cannot be applied to 3-qubit gate {}".format(gate))
    ## TODO: Add a warning or error if a bad gate name is given    

class NoiseGate: 
    def __init__(self, name):
        self.name = name
        self.channels = []

    def add_channel(self, channel):
        self.channels.append(channel)