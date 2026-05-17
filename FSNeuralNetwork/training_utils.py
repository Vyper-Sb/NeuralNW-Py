from FSNeuralNetwork.neural_network import NeuralNetwork


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
    epochs_of_epochs: list,
    repetitions: int = 1000,
    learning_rate: float = 0.01,
):

    for _ in range(repetitions):
        err: list[float] = []
        for epoch in epochs_of_epochs:
            for train_set in epoch:
                for i in range(len(train_set["X"])):
                    err.append(
                        neuralNetwork.train_with_sgd(
                            train_set["X"][i],
                            train_set["Y"][i],
                            learning_rate=learning_rate,
                        )
                    )

    err_sum: float = 0

    for error in err:
        err_sum += error

    print(f"\nNN after training: {neuralNetwork}")

    return err_sum / len(err)
