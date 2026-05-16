import random


def generate_random_weights(
    neuron_count: int, input_count: int, min_val: float = 0.001, max_val: float = 0.03
) -> list[list[float]]:
    generated_weights = []

    for _ in range(neuron_count):
        weights_neuron = []

        for _ in range(input_count):
            weights_neuron.append(round(random.uniform(min_val, max_val), 4))

        generated_weights.append(weights_neuron)

    return generated_weights


def generate_centerheavy_weights(
    neuron_count: int,
    input_count: int,
    edge_val: float = 0.001,
    center_val: float = 0.5,
) -> list[list[float]]:
    neuron_weights = []
    mid = (input_count - 1) / 2

    for i in range(input_count):
        dist_to_mid = abs(i - mid) / mid
        factor = 1 - dist_to_mid
        weight = edge_val + factor * (center_val - edge_val)
        neuron_weights.append(round(weight, 4))

    generated_weights = [neuron_weights.copy() for _ in range(neuron_count)]

    return generated_weights


def generate_edgeheavy_weights(
    neuron_count: int,
    input_count: int,
    center_val: float = 0.001,
    edge_val: float = 0.5,
) -> list[list[float]]:
    neuron_weights = []
    mid = (input_count - 1) / 2

    for i in range(input_count):
        dist_to_mid = abs(i - mid) / mid
        factor = dist_to_mid
        weight = center_val + factor * (edge_val - center_val)
        neuron_weights.append(round(weight, 4))

    generated_weights = [neuron_weights.copy() for _ in range(neuron_count)]

    return generated_weights
