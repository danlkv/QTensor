import time
import os
import numpy as np
from scipy.optimize import minimize
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from qiskit import IBMQ
from qiskit import Aer
from qiskit.visualization import plot_histogram
import qiskit
import random
import networkx
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
import copy
#--from skopt import gp_minimize
np.random.seed(42)


from qaoa_energy_only_qiskit import simulate_qiskit_amps

##Following steps for QAOA:
# 1. Create the driver and cost hamiltonians
# 2. swap runs of each for a fixed iteration (p)
# 3. optimize Beta and Gamma params
# 4. re-run until you retrieve the best Max-Cut coefficient you can


#adj_mat_example =np.matrix([[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
#p_example = 5

##this assumes nodes are just 0 through N, not any other weird order atm


def parse_args():
    parser = ArgumentParser(__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-file', action='store',default=None,
                        help='file containing adjacency matrix')
    parser.add_argument('-g', '--gamma',default=[],nargs='*', help='Gamma param')
    parser.add_argument('-b', '--beta',default=[],nargs='*', help='Beta param')
    parser.add_argument('-p', '--steps',default=5, help='Iterations of hamiltonian')
    parser.add_argument('-s', '--shots',default=1, help='Number of shots')
    parser.add_argument('-l', '--lang',default='ibm', help='output language')
    parser.add_argument('-i', '--iters',default=50, help='Nelder-Mead quantum circuit evaluation budget')
    parser.add_argument('-ER', '--Erdos',default=[],nargs=2, help='Erdos-Renyi parameters -> [num_nodes,edge_prob]')

    return parser.parse_args()


def create_graph(n_qbits,pe,file):

    if file != None:
        adj_mat = np.loadtxt(file)
        graph = networkx.from_numpy_matrix(adj_mat)
        networkx.draw(graph,with_labels=True)
        plt.savefig(file+".png")
        plt.close()
        return adj_mat,graph

    
    node_list = [x for x in range(n_qbits)]
    adj_mat = np.zeros([len(node_list),len(node_list)])
    for node_1 in node_list:
        for node_2 in node_list:
            if node_1 == node_2:
                adj_mat[node_1,node_2] = 1
            elif node_1 > node_2:
                adj_mat[node_1,node_2] = adj_mat[node_2,node_1]
            else:
                random_edge = random.uniform(0,1)
                if random_edge > pe:
                    adj_mat[node_1,node_2] = 0
                else:
                    adj_mat[node_1,node_2] = 1

                    
    graph = networkx.from_numpy_matrix(adj_mat)
    networkx.draw(graph,with_labels=True)
    plt.savefig("ER_p="+str(pe)+"_N="+str(n_qbits)+".png")
    plt.close()
    return adj_mat,graph


def random_graph(n,M):
    #graph = networkx.erdos_renyi_graph(n,.03)
    graph = networkx.random_regular_graph(3,n)

    return graph

def num_qbits(adj_mat):
    return int(adj_mat.shape[0])

def cost_ham(adj_mat,gamma,lang):
    cost_ham_circ = []
    #adj_mat is an NxN matrix with the connectivity of each node. entry i,j is whether or not nodes i and j are connected
    ##this assumes a NON-directional mapping, only looking at upper triangle
    num_nodes = num_qbits(adj_mat)
    ind = np.triu_indices(num_nodes,1)
    for x,y in zip(ind[0],ind[1]):
        if adj_mat[x,y] == 1:
            qi = str(x)
            qj = str(y)
            #gamma = str(gamma)
            double_neg_gamma = str(gamma * -2)
            if lang == 'ibm':
                cost_ham_circ.append('cx q['+qi+'],q['+qj+'];')
                cost_ham_circ.append('rz('+str(-gamma)+') q['+qj+'];')
                cost_ham_circ.append('cx q['+qi+'],q['+qj+'];')
                #cost_ham_circ.append('cu1('+double_neg_gamma+') q['+qi+'],q['+qj+'];')
                #cost_ham_circ.append('u1('+str(gamma)+') q['+qi+'];')
                #cost_ham_circ.append('u1('+str(gamma)+') q['+qj+'];')
            elif lang == 'rigetti':
                cost_ham_circ.append('CNOT '+qi+' '+qj)
                cost_ham_circ.append('RZ('+neg_gamma+') '+qj)
                cost_ham_circ.append('CNOT '+qi+' '+qj)
        
    circ_str = ''
    for ele in cost_ham_circ:
        circ_str = circ_str + ele +'\n'
        
    return cost_ham_circ,circ_str

def driver_ham(adj_mat,beta,lang):
    driver_ham_circ = []
    #basically just puts an rx gate on each qubit
    num_nodes = num_qbits(adj_mat)
    nodes = range(num_nodes)
    for x in nodes:
        qi = str(x)
        if lang == 'ibm':
            driver_ham_circ.append('rx('+str(beta*2)+') q['+qi+'];')
            #driver_ham_circ.append('h q['+qi+'];')
            #driver_ham_circ.append('rz('+str(-beta*2)+') q['+qi+'];')
            #driver_ham_circ.append('h q['+qi+'];')
        elif lang == 'rigetti':
            driver_ham_circ.append('RX('+str(2*beta)+') '+qi)
    circ_str = ''
    for ele in driver_ham_circ:
        circ_str = circ_str+ele+'\n'
    return driver_ham_circ,circ_str

#|β, γi> = VP UP · · · V2U2V1U1|ψi> 
#VP is driver, UP is cost


def circuit_gen(adj_mat,g,b,p,lang):
    #initializing hadamard gates, one on each qbit
    def barrier_make(nqbits):
        str_out = 'barrier'
        for bmq in range(nqbits):
            str_out += ' q['+str(bmq)+'],'
        str_out = str_out[:-1]
        str_out +=';\n'
        return str_out
    total_circ_str = ''
    nqbits = num_qbits(adj_mat)
    if lang == 'ibm':
        total_circ_str = total_circ_str + 'qreg q['+str(nqbits)+'];\n'
        total_circ_str = total_circ_str + 'creg c['+str(nqbits)+'];\n'
        for n in range(nqbits):
            total_circ_str = total_circ_str + 'h q['+str(n)+'];\n'
        total_circ_str = total_circ_str + barrier_make(nqbits)
    elif lang == 'rigetti':
        total_circ_str = total_circ_str + 'DECLARE ro BIT['+str(nqbits)+']\n'
        for n in range(nqbits):
            total_circ_str = total_circ_str + 'H '+str(n)+'\n'
    #driver and cost hamiltonian generation
    for steps,g,b in zip(range(p),g,b):
        cost_circ_list,cost_circ_str = cost_ham(adj_mat,g,lang)
        drive_circ_list,drive_circ_str = driver_ham(adj_mat,b,lang)
        barrier_str = barrier_make(nqbits)
        total_circ_str = total_circ_str + cost_circ_str +barrier_str+drive_circ_str+barrier_str
    for n in range(nqbits):
        if lang == 'ibm':
            total_circ_str = total_circ_str+'measure q['+str(n)+'] -> c['+str(n)+'];\n'
        elif lang == 'rigetti':
            total_circ_str = total_circ_str+'MEASURE '+str(n)+' ro['+str(n)+']\n'

    
    qasm_str_list = total_circ_str.split('\n')
    if lang == 'ibm':
        qasm_str_list.insert(0,'include "qelib1.inc";')
        qasm_str_list.insert(0,'OPENQASM 2.0;')
    meta_circ_str = ''
    for st in qasm_str_list:
        meta_circ_str = meta_circ_str + st+'\n'
    #save the circuit to file
    with open("QAOA_circ.txt",'w+') as f:
        f.write(meta_circ_str)
        f.close()
    return meta_circ_str,nqbits



def calculate_cut(adj_mat,bs_out):

    def diff_sets(s1,a,b):
        #a is in first set
        if a in s1:
            if b in s1:
                return False
            else:
                return True
        #a is in the second set
        else:
            if b in s1:
                return True
            else:
                return False

    inds = np.triu_indices(adj_mat.shape[0],1)
    s1 = []
    s2 = []
    i=0
    for bit in bs_out:
        if bit == '1':
            s1.append(i)
        else:
            s2.append(i)
        i+=1
    cut_val = 0
    for x,y in zip(inds[0],inds[1]):
        if adj_mat[x,y] == 1:
            if diff_sets(s1,x,y):
                cut_val+=1
    return cut_val

def IBM_run(meta_circ_str,nqbits,shots):

    #run the circuit, get an output bitstring
    #c = xacc.getCompiler('openqasm')
    #qasm_str = c.translate(ir_obj)
    #print(qasm_str)
    circ1 = qiskit.QuantumCircuit(nqbits,nqbits)
    circ1 = circ1.from_qasm_str(meta_circ_str)
    XACC_circ = circ1

    simulator = Aer.get_backend('qasm_simulator')
    job = qiskit.execute(XACC_circ, simulator,optimization_level=0,shots=shots,seed_simulator=10)

    #run on an actual IBM device
    #IBMQ.load_account()
    #IBMQ.providers()
    #provider = IBMQ.get_provider(group='open')
    #final_simulator_backend = provider.get_backend('ibmq_qasm_simulator')
    #job = qiskit.execute(XACC_circ, backend=final_simulator_backend,optimization_level=0,shots=shots)
    
    result = job.result()
    run_id = job.job_id()

    #fig = plot_histogram(result.get_counts(XACC_circ))
    #fig.savefig('QAOA_outs.png')
    
    
    return result.get_counts(XACC_circ)

def average_cut(bs_dict,shots,adj_mat):
    #This will take a look at the average cut of the run, used for parameter tuning

    good_res = shots*.01
    cut_sum = 0
    shots_sum = 0
    for res in bs_dict:
        num_shots = int(bs_dict[res])
        #for implementing the 1% filter
        #if num_shots > 1:
        for x in range(num_shots):
            res_test = res[::-1]
            cut_sum += calculate_cut(adj_mat,res_test)
        shots_sum += int(bs_dict[res])
    avg_cut = cut_sum / shots_sum
    return avg_cut

def final_cut(bs_dict,adj_mat):
    #final cut is the maximum of all the individual cuts found in the final bs output
    #for now this will test ANY bitstring that comes up in the resultsx
    fin_cut = 0
    res_f = ''
    for res in bs_dict:
        #flipping bitstring out to match expected node values
        res_test = res[::-1]
        loc_cut = calculate_cut(adj_mat,res_test)
        if loc_cut > fin_cut:
            fin_cut = loc_cut
            res_f = res
    return fin_cut,res_f

def run_func(x0,adj_mat,p,lang,shots):
    #pass this function through classical optimization
    assert len(x0)==2*p
    g = x0[:len(x0)//2]
    b = x0[len(x0)//2:]
    meta_circ_str,nqbits = circuit_gen(adj_mat,g,b,p,lang)
    run = IBM_run(meta_circ_str,nqbits,shots)
    #all bitstrings need to be flipped
    avg_cut = average_cut(run,shots,adj_mat)
    if avg_cut == 0:
        print(x0)
        print('stop')
        raise ValueError
    return -avg_cut if avg_cut!=0 else 0


def main(file=None,p=5,shots=1,g=[],b=[],lang='ibm',opt_iters=50,ER=[]):

    shots = int(shots)
    p = int(p)
    opt_iters = int(opt_iters)

    if len(g) != 0:
        g = [float(x) for x in g]
    if len(b) != 0:
        b = [float(x) for x in b]
    
    print('')
    print('QAOA with p='+str(p)+', shots='+str(shots))
    print('')



    
    if len(g) == 0:# and len(b) > 0:
        for it in range(p):
            g.append(np.random.normal(.5,.01))
    if len(b) == 0:# and len(g) > 0:
        for it in range(p):
            b.append(np.random.normal(.5,.01))
    x0 = np.array(g+b)

    def simulator_call(a,b):
        return 0

    #adj_mat,p,lang,shots
    def run_func_scaled(x0):
        #pass this function through classical optimization
        g = x0[:len(x0)//2]
        b = x0[len(x0)//2:]
        #feed graph into simulator
        ##run = INSERT HERE##
        meta_circ_str,nqbits = circuit_gen(meta_adj_mat,g,b,p,'ibm')
        run = IBM_run(meta_circ_str,nqbits,shots)
        #need to check that run output structure will match the average_cut function#
        avg_cut = average_cut(run,shots,meta_adj_mat)
        return -avg_cut if avg_cut!=0 else 0
    def run_func_final(x):
        assert len(x)==2*p
        g_meta = x[:len(x)//2]
        b_meta = x[len(x)//2:]
        
        meta_circ_str,nqbits = circuit_gen(meta_adj_mat,g_meta,b_meta,p,'ibm')
        meta_run = IBM_run(meta_circ_str,nqbits,shots)
        meta_avg_cut = average_cut(meta_run,shots,meta_adj_mat)
        fin_cut,bs = final_cut(meta_run,meta_adj_mat)
        return meta_avg_cut,fin_cut,meta_run,bs,meta_circ_str

    def scale_down_prob(e_l,N_l,N_s):
        return (e_l/N_l)*N_s

    try:
        os.mkdir('large_graphs')
    except:
        pass
    try:
        os.mkdir('scaled_down_graphs')
    except:
        pass
    try:
        os.mkdir('results')
    except:
        pass

    num_large_graphs = 1
    num_scaled_graphs = 3

    def cost_function_C(x,G):
        E = G.edges()
        if( len(x) != len(G.nodes())):
            return np.nan
        C = 0;
        for index in E:
            e1 = index[0]
            e2 = index[1]
            w  = 1
            C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
        return C



    from qensor import QAOA_energy, QtreeQAOAComposer, QtreeSimulator
    from qensor import QAOA_energy_no_lightcones
    from qensor import CirqQAOAComposer, CirqSimulator
    from qensor.ProcessingFrameworks import PerfNumpyBackend

    ##TESTING STARTS HERE##

    def simulate_one_amp(G, gamma, beta):
        composer = QtreeQAOAComposer(
            graph=G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        print(composer.circuit)
        sim = QtreeSimulator()
        result = sim.simulate(composer.circuit)
        print('Qensor 1 amp',result.data)
        print('Qensor 1 prob',np.abs(result.data)**2)


    def profile_graph(G, gamma, beta):
        
        #print(G)
        meta_adj_mat = networkx.to_numpy_array(G)
        print(meta_adj_mat)
        gamma, beta = np.array(gamma), np.array(beta)
        #print(meta_adj_mat)

        start = time.time()
        def simulate_qiskit(meta_adj_mat, gamma, beta):
            meta_circ_str, nqbits = circuit_gen(meta_adj_mat,gamma,beta,p,'ibm')
            print('qiskit cirq\n', meta_circ_str)
            meta_run = IBM_run(meta_circ_str,nqbits,shots)
            print('qiskit sim time', time.time()-start)

            #print(meta_run)
            avr_C=0
            max_C  = [0,0]
            hist   = {}
            for k in range(len(G.edges())+1):
                hist[str(k)] = hist.get(str(k),0)

            for sample in list(meta_run.keys()):

                # use sampled bit string x to compute C(x)
                x         = [int(num) for num in reversed(list(sample))]
                #x         = [int(num) for num in (list(sample))]
                tmp_eng   = cost_function_C(x,G)
                #print("cost", x, tmp_eng)
            
                # compute the expectation value and energy distribution
                avr_C     = avr_C    + meta_run[sample]*tmp_eng
                hist[str(round(tmp_eng))] = hist.get(str(round(tmp_eng)),0) + meta_run[sample]
            
                # save best bit string
                if( max_C[1] < tmp_eng):
                    max_C[0] = sample
                    max_C[1] = tmp_eng
            print(hist)

            qiskit_time = time.time() - start
            label = '0'*G.number_of_nodes()
            try:
                print('Qiskit first prob: ',meta_run[label]/shots)
            except KeyError:
                print('Qiskit does not have samples for state 0')
            return qiskit_time, avr_C/shots
        try:
            qiskit_time, qiskit_e = 0, 0
            qiskit_time, qiskit_e = simulate_qiskit(meta_adj_mat, gamma, beta)
        except Exception as e:
            print('Qiskit error', e)

        #gamma, beta = [-gamma/2/np.pi, gamma, gamma], [beta/1/np.pi, beta, beta]
        qiskit_result = simulate_qiskit_amps(G, gamma, beta)


        gamma, beta = -gamma/2/np.pi, beta/1/np.pi

        start = time.time()
        print(gamma, beta)
        E = QAOA_energy(G, gamma, beta, profile=False)
        #assert E-E_no_lightcones<1e-6

        qensor_time = time.time() - start
        print('\n Qensor:', E)

        print('####== Qiskit:', qiskit_e)
        print('Delta with qensor:',E-qiskit_e)

        print('####== Qiskit amps result', qiskit_result)
        print('Delta with qensor:',E-qiskit_result)

        print('qiskit energy time', qiskit_time)
        print('Qensor full time', qensor_time)
        assert abs(E-qiskit_result) < 1e-6, 'Results of qensor do not match with qiskit'

        return qiskit_time, qensor_time

    gamma, beta = [np.pi/8], [np.pi/6]
    qiskit_profs = []
    qensor_profs = []
    graphs = [networkx.random_regular_graph(3, n) for n in range(4, 16, 2)]
    for n in range(4, 8, 1):
        G = networkx.complete_graph(n)
        #graphs.append(G)

        continue
        G = networkx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(range(n), range(1,n)))
        G.add_edges_from([[0,n-1]])
        graphs.append(G)
    if False:
        n= 4
        G = networkx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(range(n), range(1,n)))
        G.add_edges_from([[0,n-1]])
        graphs.append(G)
        #graphs = []

        elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
        G = networkx.OrderedGraph()
        G.add_edges_from(elist)
        graphs.append(G)

    for G in graphs:
        #G.add_edges_from([[1,n-1]])

        qiskit_time, qensor_time = profile_graph(G, gamma, beta)
        qiskit_profs.append(qiskit_time)
        qensor_profs.append(qensor_time)

        #print((G.number_of_edges()-E)/2)
    tostr = lambda x: [str(a) for a in x]
    print(', '.join(tostr(qiskit_profs)))
    print(', '.join(tostr(qensor_profs)))



if __name__ == '__main__':
    args = parse_args()
    main(args.file,args.steps,args.shots,args.gamma,args.beta,args.lang,args.iters,args.Erdos)
