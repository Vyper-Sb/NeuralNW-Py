from FSNeuralNetwork.neural_layers import (
    InputNeuralLayer,
    HiddenNeuralLayer,
    OutputNeuralLayer,
)
from typing import Optional
from FSNeuralNetwork.activation_functions import activationType
import json


class NeuralNetwork:
    def __init__(
        self,
        inputs: int,
        outputs: int,
        output_bias: float = -0.5,
        output_activation_type: activationType = activationType.SIGMOID,
        output_weights: Optional[list[list[float]]] = None,
        output_alpha: float = 1,
    ) -> None:
        self.hiddenLayers: list[HiddenNeuralLayer] = []
        self.inputsOfLastActiveLayer: int = inputs
        self.inputLayer: InputNeuralLayer = InputNeuralLayer(inputs_count=inputs)
        self.outputLayer: OutputNeuralLayer = OutputNeuralLayer(
            outputs,
            inputs,
            bias=output_bias,
            activation=output_activation_type,
            weights=output_weights,
            alpha=output_alpha,
        )

    def add_hidden_layer(
        self,
        neurons: int,
        bias: float = -0.5,
        activation_type: activationType = activationType.RELU,
        weights: Optional[list[list[float]]] = None,
        alpha: float = 1,
    ) -> None:
        hiddenLayer = HiddenNeuralLayer(
            neurons,
            self.inputsOfLastActiveLayer,
            bias=bias,
            activation=activation_type,
            weights=weights,
            alpha=alpha,
        )
        self.inputsOfLastActiveLayer = neurons
        self.hiddenLayers.append(hiddenLayer)
        self.outputLayer.set_inputs(neurons)

    def calculate_data(self, data: list[float]) -> list[float]:
        self.inputLayer.set_inputs(data)
        layer_out = self.inputLayer.forward()
        # print(f"Output Input Layer: {layer_out}")

        for hiddenLayer in self.hiddenLayers:
            hidden_layer_out = hiddenLayer.forward(layer_out)
            # print(f"Output Hidden Layer: {hidden_layer_out}")
            layer_out = hidden_layer_out

        result = self.outputLayer.forward(layer_out)
        # print(f"Result after Output Layer: {result}")
        return result

    def get_weights_and_biases(self) -> dict:
        hidden_layers_wb = []

        for hiddenLayer in self.hiddenLayers:
            hidden_layers_wb.append(hiddenLayer.return_weights_and_biases)

        output_layer_wb = self.outputLayer.return_weights_and_biases

        return {
            "hidden_layers_wb": hidden_layers_wb,
            "output_layer_wb": output_layer_wb,
        }

    def save_neural_network(self, file_path: str = "neural_network_config.json"):
        input_layer_config: dict = self.inputLayer.return_config()
        hidden_layer_configs: list[dict] = []
        for hiddenLayer in self.hiddenLayers:
            hidden_layer_configs.append(hiddenLayer.return_config())

        output_layer_config: dict = self.outputLayer.return_config()

        neural_network_config = {
            "input_layer": input_layer_config,
            "hidden_layers": hidden_layer_configs,
            "output_layer": output_layer_config,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(neural_network_config, f, indent=4)

    @classmethod
    def load_neural_network(cls, file_path: str = "neural_network_config.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            out_cfg = data["output_layer"]

            network = cls(
                inputs=data["input_layer"]["inputs"],
                outputs=out_cfg["neurons"],
                output_bias=-0.5,
                output_activation_type=activationType(out_cfg["activation"]),
                output_alpha=out_cfg.get("alpha", 1),
            )

            network.outputLayer.set_biases(out_cfg["biases"])

            for hidden_layer_cfg in data.get("hidden_layers", []):
                network.add_hidden_layer(
                    neurons=hidden_layer_cfg["neurons"],
                    bias=-0.5,
                    activation_type=activationType(hidden_layer_cfg["activation"]),
                    weights=hidden_layer_cfg["weights"],
                    alpha=hidden_layer_cfg.get("alpha", 1),
                )
                network.hiddenLayers[-1].set_biases(hidden_layer_cfg["biases"])

            network.outputLayer.set_weights(
                out_cfg["weights"],
            )

            print(f"loaded_nn: {network}")

            return network
        except Exception as e:
            raise Exception(f"Error: {e}")

    def train_with_sgd(
        self,
        batch: list[list[float]],
        batch_target_output: list[list[float]],
        learning_rate: float,
    ) -> tuple[float, list[list[float]]]:
        # 1.Forward Pass
        total_loss = 0.0
        all_predicted_outputs: list[list[float]] = []

        for training_input, target_output in zip(batch, batch_target_output):
            predicted_output = self.calculate_data(training_input)
            all_predicted_outputs.append(predicted_output)

            output_errors: list[float] = []

            for j in range(len(predicted_output)):
                error = predicted_output[j] - target_output[j]
                total_loss += 0.5 * error**2
                output_errors.append(error)

            current_errors = self.outputLayer.backward(output_errors, learning_rate)

            for hiddenLayer in reversed(self.hiddenLayers):
                current_errors = hiddenLayer.backward(current_errors, learning_rate)

        average_loss = total_loss / len(batch)

        return average_loss, all_predicted_outputs

    def __str__(self) -> str:
        structure = f"NeuralNetwork Architektur:\n"
        structure += f"  [Eingang]  {self.inputLayer}\n"
        for i, layer in enumerate(self.hiddenLayers):
            structure += f"  [Hidden {i+1}] {layer}\n"
        structure += f"  [Ausgang]  {self.outputLayer}"
        return structure

    def __repr__(self) -> str:
        return (
            f"NeuralNetwork(inputs={self.inputLayer.inputs_count}, "
            f"hidden_layers_count={len(self.hiddenLayers)}, "
            f"outputs={self.outputLayer.neurons_count})"
        )
