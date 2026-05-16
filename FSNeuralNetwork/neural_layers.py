from FSNeuralNetwork.neural_network_basic_components import NeuralLayer


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

    def __str__(self) -> str:
        neuron_strings = "\n  ".join(
            [f"[{i}]: {str(n)}" for i, n in enumerate(self.Neurons)]
        )
        return (
            f"Output NeuralLayer mit {len(self.Neurons)} Neuronen:\n  {neuron_strings}"
        )

    def __repr__(self) -> str:
        return f"OutputNeuralLayer(neurons_count={len(self.Neurons)}, weights={self.weights}, neurons={self.Neurons})"
