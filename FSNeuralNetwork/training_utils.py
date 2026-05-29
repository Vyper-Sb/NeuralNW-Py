from FSNeuralNetwork.neural_network import NeuralNetwork
import numpy as np


def train_NEpochs_alternately(
    neuralNetwork: NeuralNetwork,
    data: list,
    epochs: int = 3,
    learning_rate: float = 0.01,
):
    data_arr = np.array(data, dtype=object)
    num_batches = len(data_arr)

    for i in range(epochs):
        print(f"--- Epoche {i + 1}/{epochs} ---")
        for j, batch in enumerate(data_arr):
            print("batch ", j + 1, "/", num_batches)
            input_batch = batch["X"]
            target_batch = batch["Y"]
            avg_loss, _ = neuralNetwork.train_with_sgd(
                batch=input_batch,
                batch_target_output=target_batch,
                learning_rate=learning_rate,
            )
            print(avg_loss)

        np.random.shuffle(data_arr)
