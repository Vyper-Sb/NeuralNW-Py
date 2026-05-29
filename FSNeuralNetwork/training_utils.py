from FSNeuralNetwork.neural_network import NeuralNetwork
import numpy as np

def train_neuralnetwork(
    neuralNetwork: NeuralNetwork,
    XTrain: list[list[float]],
    YTrain: list[list[float]],
    repetitions: int = 1000,
    learning_rate: float = 0.01,
) -> float:
    for _ in range(repetitions):
        for i in range(len(XTrain)):
            err = neuralNetwork.train_with_sgd(
                XTrain[i], YTrain[i], learning_rate=learning_rate
            )

    return err


def train_nn_with_epochs(
    neuralNetwork: NeuralNetwork,
    epochs_data: list,
    repetitions: int = 1000,
    learning_rate: float = 0.01,
):

    for _ in range(repetitions):
        err: list[float] = []
        for epoch in epochs_data:
            for i in range(len(epoch["X"])):
                err.append(
                    neuralNetwork.train_with_sgd(
                        epoch["X"][i], epoch["Y"][i], learning_rate=learning_rate
                    )
                )

    err_sum: float = 0

    for error in err:
        err_sum += error

    print(f"\nNN after training: {neuralNetwork}")

    return err_sum / len(err)


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
            print("batch ", j+1 ,"/",num_batches)
            input_batch = batch["X"]
            target_batch = batch["Y"]
            avg_loss, _ = neuralNetwork.train_with_sgd(
                batch=input_batch,
                batch_target_output=target_batch,
                learning_rate=learning_rate,
            )
            print(avg_loss)
        
        np.random.shuffle(data_arr)
                    

    
