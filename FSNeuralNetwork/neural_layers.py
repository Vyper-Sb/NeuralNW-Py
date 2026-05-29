from FSNeuralNetwork.neural_network_basic_components import NeuralLayer
from FSNeuralNetwork.loss_functions import LossType


class InputNeuralLayer:
    def __init__(self, inputs_count: int) -> None:
        self.inputs_count = inputs_count
        self.current_values = []  # Speichert die aktuellen Eingabedaten

    def set_inputs(self, values: list[float]) -> None:
        """Nimmt die Rohdaten entgegen und prüft die Dimension."""
        if len(values) != self.inputs_count:
            raise ValueError(
                f"Dimension falsch! Erwartet: {self.inputs_count} Werte, "
                f"Erhalten: {len(values)} Werte."
            )
        self.current_values = values

    def forward(self) -> list[float]:
        """Gibt die Daten einfach unverändert weiter."""
        return self.current_values

    def return_config(self) -> dict:
        return {"inputs": self.inputs_count}

    def __str__(self) -> str:
        return f"InputNeuralLayer mit {self.inputs_count} Eingängen."

    def __repr__(self) -> str:
        return f"InputNeuralLayer(inputs_count={self.inputs_count})"


class HiddenNeuralLayer(NeuralLayer):
    def __init__(self, neurons: int, inputs: int, **kwargs) -> None:
        super().__init__(neurons=neurons, inputs=inputs, **kwargs)

    def __str__(self) -> str:
        neuron_strings = "\n  ".join(
            [f"[{i}]: {str(n)}" for i, n in enumerate(self.Neurons)]
        )
        return (
            f"HiddenNeuralLayer mit {len(self.Neurons)} Neuronen:\n  {neuron_strings}"
        )

    def __repr__(self) -> str:
        return f"HiddenNeuralLayer(neurons_count={len(self.Neurons)}, weights={self.weights}, neurons={self.Neurons})"


class OutputNeuralLayer(NeuralLayer):
    def __init__(self, neurons: int, inputs: int, **kwargs) -> None:
        super().__init__(neurons=neurons, inputs=inputs, **kwargs)
    

    def backward(self, errs_right: list[float], learning_rate: float, loss_type:LossType) -> list[float]:

        old_weights = [neuron.weights.copy() for neuron in self.Neurons]
        deltas: list[float] = []

        use_direct_delta = loss_type in [
            LossType.BINARY_CROSS_ENTROPY,
            LossType.CATEGORICAL_CROSS_ENTROPY,
            LossType.SPARSE_CATEGORICAL_CROSS_ENTROPY,
        ]

        if not(use_direct_delta):
            derivative_activationFunc = self.activation.get_activationFunc_derivative()

            for i, neuron in enumerate(self.Neurons):
                delta = neuron.backward(
                    err_right=errs_right[i],
                    learning_rate=learning_rate,
                    derivative_activationFunc=derivative_activationFunc,
                )
                deltas.append(delta)
        
        else: 
            for i, neuron in enumerate(self.Neurons):
                delta = errs_right[i]

                if neuron.last_input is None:
                    raise ValueError("Forward pass must run before backward pass.")

                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * delta * neuron.last_input[j]

                neuron.bias -= learning_rate * delta


                deltas.append(delta)


        errors_for_left = [0.0] * self.inputs_count
        #print("old_weights:", old_weights)

        for j in range(self.inputs_count):
            error_sum = 0.0
            for k  in range (self.neurons_count):

                error_sum += deltas[k] * old_weights[k][j]

            errors_for_left[j] = error_sum

        self.weights = [neuron.weights for neuron in self.Neurons]
        #print("new weights:", self.weights)
      
        return errors_for_left


    def __str__(self) -> str:
        neuron_strings = "\n  ".join(
            [f"[{i}]: {str(n)}" for i, n in enumerate(self.Neurons)]
        )
        return (
            f"Output NeuralLayer mit {len(self.Neurons)} Neuronen:\n  {neuron_strings}"
        )

    def __repr__(self) -> str:
        return f"OutputNeuralLayer(neurons_count={len(self.Neurons)}, weights={self.weights}, neurons={self.Neurons})"
