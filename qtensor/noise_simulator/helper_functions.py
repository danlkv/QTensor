from collections import OrderedDict
import numpy as np
import numpy as np
from numpy import inner
from numpy.linalg import norm
import json
import jsbeautifier
import networkx as nx

import qtensor

def decimal_to_binary(n):
# converting decimal to binary
# and removing the prefix(0b)
    return bin(n).replace("0b", "")

def create_counts_dict(num_qubits, big_endian):
    # get the binary length of num_qubits
    length = int(np.ceil(np.log(2**num_qubits + 1)/np.log(2)) - 1)
    counts = OrderedDict()
    for i in range(2**num_qubits):
        # convert to binary and then pad the right side with 0s
        # length is used so that the length of each key is the same
        if big_endian == True:
            key_i = str(decimal_to_binary(i).zfill(length))
            key_i[::-1]
            counts[key_i] = 0
        else:
            key_i = str(decimal_to_binary(i).zfill(length))
            counts[key_i] = 0
    #print(counts)
    return counts

# big_endian refers to the order that qubits are displayed 
# Use big_endian = False if you want the qubit ordering that Qiskit uses 
def attach_qubit_names(probs_list, big_endian = True):
    num_qubits = int(np.log2(len(probs_list)))
    probs_dict = create_counts_dict(num_qubits, big_endian)
    for key, prob in zip(probs_dict, probs_list):
        probs_dict[key] = prob
    return probs_dict

def fidelity(A, B):
    return (np.sum(np.diagonal(np.sqrt(np.sqrt(A)*B*np.sqrt(A)))))**2

def cosine_similarity(A, B):
    return inner(A, B) / (norm(A)* norm(B)) 

def get_qaoa_params(n, p, d):
    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even.")
    if not 0 <= d < n:
        raise ValueError("Bad value for d. The inequality 0 <= d < n must be satisfied.")

    print('Circuit params: n: {}, p: {}, d: {}'.format(n, p, d))
    G = nx.random_regular_graph(d, n)
    gammabeta = np.array(qtensor.tools.BETHE_QAOA_VALUES[str(d)]['angles'])
    #gammabeta = gammabeta.tolist()
    gamma = gammabeta[:d]
    beta = gammabeta[d:]
    gamma = gamma.tolist()
    beta = beta.tolist()
    #gamma = [np.random.uniform(0, 2*np.pi) for _ in range(p)]
    #beta = [np.random.uniform(0, np.pi) for _ in range(p)]
    return G, gamma, beta

def save_dict_to_file(dict, name):
    options = jsbeautifier.default_options()
    options.indent_size = 2
    with open(name, 'a') as f:
        f.write(jsbeautifier.beautify(json.dumps(dict), options))

def get_total_jobs(num_circs: int, num_nodes: int, num_jobs_per_node: int, min_circs_per_job = 10):
    ## We make sure that each job has a minimum number of circuits. We do this because 
    ## if there are too few circuits for each unit of work, the overhead from 
    ## parallelization removes any advantage gained 
    ## TODO: determine a better minimum number of circuits. Currently 10 is chosen arbitrarily
    min_circs_per_job = min(min_circs_per_job, num_circs)
    if num_nodes * num_jobs_per_node > num_circs / min_circs_per_job:
        num_circs_per_job = min_circs_per_job
        total_jobs = int(np.ceil(num_circs / num_circs_per_job))
    else: 
        total_jobs = num_nodes * num_jobs_per_node
        num_circs_per_job = int(np.ceil(num_circs / total_jobs))
    return total_jobs, num_circs_per_job


if __name__ == '__main__':
    G, gamma, beta = get_qaoa_params(3,2,2)