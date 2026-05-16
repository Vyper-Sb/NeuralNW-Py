import FSNeuralNetwork.neural_network as neural_network

network = neural_network.NeuralNetwork(16, 5)

network.add_hidden_layer(100)
network.add_hidden_layer(100)

data = [
    0.45,
    0.39,
    0.2,
    0.9,
    0.12,
    0.34,
    0.21,
    0.8,
    0.6,
    0.767,
    0.08,
    0.69,
    0.42,
    0.512,
    1.4,
    0.1,
]
output = network.calculate_data(data)
print(output)

network.save_neural_network()
