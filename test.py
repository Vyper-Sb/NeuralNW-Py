from FSNeuralNetwork.neural_network import *
from FSNeuralNetwork.activation_functions import activationType


network = NeuralNetwork.load_neural_network("test.json")

for i in range(1):
    network.train_with_sgd([[2]], [[1]], 0.1)

network.save_neural_network("weights_test2.json")

#print(network.calculate_data([1]))