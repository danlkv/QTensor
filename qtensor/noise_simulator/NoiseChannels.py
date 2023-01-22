import itertools as it

class DepolarizingChannel: 
    """
    In a depolarizing channel, there is a depolarizing parameter lambda
    which is given by λ = 4^n / (4^n - 1) , where n is the number of qubits
    when we run this function, we pass in a parameter term param, where 
    0 <= param <= λ. The probability of an error occuring is related to param and λ by
    prob_depol_error = param/λ, and the probability of an particular pauli error 
    from the depolarizing channel is given by param / 4^n
    """

    def __init__(self, param, num_qubits):

        num_terms = 4**num_qubits
        lam = num_terms / (num_terms - 1)
        
        if param < 0:
            raise ValueError("Error param is too small. It cannot be less than 0")
        elif param > lam:
            raise ValueError("Error param  is too large. It cannot be greater than {} when the channel has {} qubits").format(lam, num_qubits)

        self.name = 'depolarizing'
        self.param = param
        self.num_qubits = num_qubits
        self.num_terms = num_terms
        self.lam = lam
        self.kraus_ops = []
        self.qubits = []
        self._get_channel()

    def _get_channel(self):
        pauli_error_prob = self.param / self.num_terms
        identity_prob = 1 - self.param / self.lam
        depol_channel_probs = [identity_prob] + (self.num_terms - 1) * [pauli_error_prob]

        # it.product creates a set of tuples, where each element is a 
        # cartensian product of the num_qubit and each of the pauli operators. 
        # We use list(tup) for tup in beforehand to return a list instead of a tuple
        depol_channel_paulis = (''.join(tup) for tup in it.product('IXYZ', repeat=self.num_qubits))
        depol_channel = list(zip(depol_channel_paulis, depol_channel_probs))
        self.kraus_ops.extend(depol_channel)

    def add_qubits(self, qubits = ['all']):
        if qubits is not None and not isinstance(qubits, list):
            raise TypeError("qubits parameter must either be a list or left blank")

        self.qubits.extend(qubits)

    
class BitFlipChannel:
    def __init__(self, param):
        
        if param < 0:
            raise ValueError("Error param is too small. It cannot be less than 0")
        elif param > 1:
            raise ValueError("Error param is too large. It cannot be greater than 1")
        
        self.name = 'bit_flip'
        self.param = param
        self.num_qubits = 1
        self.kraus_ops = []
        self.qubits = []
        self._get_channel()

    def _get_channel(self):
        bit_flip_prob = self.param 
        identity_prob = 1 - self.param
        bit_flip_channel = [('I', identity_prob), ('X', bit_flip_prob)]
        self.kraus_ops.extend(bit_flip_channel)
    
    def add_qubits(self, qubits = ['all']):
        if qubits is not None and not isinstance(qubits, list):
            raise TypeError("qubits parameter must either be a list or left blank")
        if len(qubits) > 1:
            raise Exception("Bit Flip channel is only defined for a single qubit")

        self.qubits.extend(qubits)

class PhaseFlipChannel:
    def __init__(self, param):
        
        if param < 0:
            raise ValueError("Error param is too small. It cannot be less than 0")
        elif param > 1:
            raise ValueError("Error param is too large. It cannot be greater than 1")
        
        self.name = 'phase_flip'
        self.param = param
        self.num_qubits = 1
        self.kraus_ops = []
        self.qubits = []
        self._get_channel()

    def _get_channel(self):
        phase_flip_prob = self.param 
        identity_prob = 1 - self.param
        phase_flip_channel = [('I', identity_prob), ('Z', phase_flip_prob)]
        self.kraus_ops.extend(phase_flip_channel)
        

    def add_qubits(self, qubits = ['all']):
        if qubits is not None and not isinstance(qubits, list):
            raise TypeError("qubits parameter must either be a list or left blank")
        if len(qubits) > 1:
            raise Exception("Phase Flip channel is only defined for a single qubit")

        self.qubits.extend(qubits)

class KrausOperator:
    def __init__(self, name, prob):
        self.name = name
        self.prob = prob