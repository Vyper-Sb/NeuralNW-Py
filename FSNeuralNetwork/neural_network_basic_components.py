from collections.abc import Callable
from typing import Optional
from FSNeuralNetwork.activation_functions import activationType, ReLU, softmax


class Neuron:
    def __init__(
        self,
        weights: list[float],
        bias=-0.5,
        activationFunc: Optional[Callable[..., float]] = ReLU,
        alpha: float = 1.0,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activationFunc = activationFunc
        self.alpha = alpha
        self.last_input: Optional[list[float]] = None
        self.last_total_value: Optional[float] = None
        self.last_output: Optional[float] = None

    def calculate_totalvalue(self, inputs: list[float]) -> float:
        self.last_input = inputs
        input_sum = 0

        for i, inp in enumerate(inputs):
            input_sum += inp * self.weights[i]

        self.last_total_value = input_sum + self.bias

        return self.last_total_value

    def calculate_output(self, inputs: list[float]) -> float:
        total_value = self.calculate_totalvalue(inputs)

        if self.activationFunc is None:
            self.last_output = total_value
            return total_value

        self.last_output = self.activationFunc(total_value, self.alpha)
        return self.last_output

    def backward(
        self,
        err_right: float,
        learning_rate: float,
        derivative_activationFunc: Callable[..., float],
    ) -> float:

        if self.last_input is None or self.last_total_value is None:
            raise ValueError(
                "No data of last input or last total value (sum+bias) is existing. Try running a forwarding before running the backwarding"
            )

        delta = err_right * derivative_activationFunc(self.last_total_value, self.alpha)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * self.last_input[i]

        self.bias -= learning_rate * delta

        return delta

    def __str__(self) -> str:
        return f"Neuron(Gewichte: {self.weights}, Bias: {self.bias})"

    def __repr__(self) -> str:
        func_name = getattr(self.activationFunc, "__name__", str(self.activationFunc))
        return f"Neuron(weights={self.weights}, bias={self.bias}, activationFunc={func_name}, alpha={self.alpha})"


class NeuralLayer:
    def __init__(
        self,
        neurons: int,  # Amount of neurons
        inputs: int,  # Amount of Inputs to every neuron
        bias: float = -0.5,  # bias of every neuron
        activation: activationType = activationType.RELU,  # activation function for every neuron
        weights: Optional[
            list[list[float]]
        ] = None,  # weights for every connecten to every neuron
        alpha: float = 1,  # alpha value for every neuron if the ectivation function ELU is used
    ) -> None:
        self.neurons_count = neurons
        self.inputs_count = inputs
        self.activation = activation
        self.alpha = alpha

        self.Neurons: list[Neuron] = []
        activationFunc = self.activation.get_activationFunc()

        for _ in range(self.neurons_count):
            neuron = Neuron(
                weights=[0.5] * self.inputs_count,
                bias=bias,
                activationFunc=activationFunc,
                alpha=self.alpha,
            )
            self.Neurons.append(neuron)

        self.set_weights(weights)

    def set_inputs(self, new_inputs: int) -> None:
        self.inputs_count = new_inputs
        default_weights = [[0.5] * new_inputs for _ in range(self.neurons_count)]
        self.set_weights(default_weights)

    def set_weights(self, weights: Optional[list[list[float]]]) -> None:
        if weights is None:
            if (
                hasattr(self, "weights")
                and len(self.weights) == self.neurons_count
                and all(len(w) == self.inputs_count for w in self.weights)
            ):
                print("current weights are suiting. No changes needed")
                return
            print("No weights set: Filling up with 0.5")
            weights = [[0.5] * self.inputs_count for _ in range(self.neurons_count)]

        elif len(weights) == self.neurons_count and all(
            len(w) == self.inputs_count for w in weights
        ):
            pass

        else:
            print(
                "Dimensions of weight were not correct. Missing weights will be filled with 0 and remaining weights will be cut."
            )
            adjusted_weights = []
            for i in range(self.neurons_count):

                if i < len(weights) and weights[i] is not None:
                    neuron_weights = weights[i]
                else:
                    neuron_weights = []

                adjusted = [
                    neuron_weights[j] if j < len(neuron_weights) else 0.0
                    for j in range(self.inputs_count)
                ]
                adjusted_weights.append(adjusted)

            weights = adjusted_weights

        self.weights = weights
        for i, neuron in enumerate(self.Neurons):
            neuron.weights = weights[i]

    def set_biases(self, biases: list[float]) -> None:
        if len(biases) != self.neurons_count:
            raise ValueError(
                f"Anzahl der Biases ({len(biases)}) stimmt nicht mit Neuronenanzahl ({self.neurons_count}) überein."
            )

        for i, neuron in enumerate(self.Neurons):
            neuron.bias = biases[i]

    @property
    def biases(self) -> list[float]:
        """Gibt die aktuellen Biases aller Neuronen als Liste zurück."""
        return [neuron.bias for neuron in self.Neurons]

    def forward(self, inputs: list[float]) -> list[float]:

        layer_outputs = [
            neuron.calculate_output(inputs=inputs) for neuron in self.Neurons
        ]

        if self.activation == activationType.SOFTMAX:
            return softmax(layer_outputs)

        return layer_outputs

    def backward(self, errs_right: list[float], learning_rate: float) -> list[float]:
        derivative_activationFunc = self.activation.get_activationFunc_derivative()

        deltas: list[float] = []
        for i, neuron in enumerate(self.Neurons):
            delta = neuron.backward(
                err_right=errs_right[i],
                learning_rate=learning_rate,
                derivative_activationFunc=derivative_activationFunc,
            )
            deltas.append(delta)

        errors_for_left = [0.0] * self.inputs_count

        for j in range(self.inputs_count):
            error_sum = 0.0
            for k, neuron in enumerate(self.Neurons):

                error_sum += deltas[k] * neuron.weights[j]

            errors_for_left[j] = error_sum

        self.weights = [neuron.weights for neuron in self.Neurons]

        return errors_for_left

    def return_weights_and_biases(self) -> dict[str, list]:
        return {"weights": self.weights, "biases": self.biases}

    def return_config(self) -> dict:
        return {
            "neurons": self.neurons_count,
            "inputs": self.inputs_count,
            "biases": self.biases,
            "activation": self.activation.value,
            "weights": self.weights,
            "alpha": self.alpha,
        }
