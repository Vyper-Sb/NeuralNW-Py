from FSNeuralNetwork.neural_network import *
from FSNeuralNetwork.activation_functions import activationType


network = NeuralNetwork.load_neural_network("test.json")

for i in range(100):
    network.train_with_sgd([[1]], [[2]], 0.5)

network.save_neural_network("test_after_train-100.json")

print(network.calculate_data([1]))