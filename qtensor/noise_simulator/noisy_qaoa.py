from NoiseChannels import *
from NoiseSimulator import * 
from NoiseModel import *
from NoiseSimComparisonResult import *
from helper_functions import *
from QAOANoiseSimulator import *

prob_1 = 0.01
prob_2 = 0.1

# QTensor Noise Model
depol_chan_1Q = DepolarizingChannel(prob_1, 1)
depol_chan_2Q = DepolarizingChannel(prob_2, 2)

noise_model = NoiseModel()
noise_model.add_channel_to_all_qubits(depol_chan_1Q, ['H', 'XPhase', 'ZPhase'])
noise_model.add_channel_to_all_qubits(depol_chan_2Q, ['cX'])

qaoa_sim = QAOANoiseSimulator()
qaoa_sim.noisy_qaoa_sim(n = 6, p = 1, d = 1, num_circs = 100, noise_model = noise_model)
#print(qaoa_sim.probs)