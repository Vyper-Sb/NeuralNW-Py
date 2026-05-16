import math
from enum import StrEnum


def ReLU(totalvalue: float, *args) -> float:
    return max(totalvalue, 0)


def ReLU_derivative(totalvalue: float, *args) -> float:
    return 1.0 if totalvalue > 0 else 0.0


def LeakyReLU(totalvalue: float, *args) -> float:
    return totalvalue if totalvalue > 0 else 0.01 * totalvalue


def LeakyReLU_derivative(totalvalue: float, *args) -> float:
    return 1.0 if totalvalue > 0 else 0.01


def sigmoid(totalvalue: float, *args) -> float:
    return 1 / (1 + math.exp(-totalvalue))


def sigmoid_derivative(totalvalue: float, *args) -> float:
    # Mathematische Formel: sigmoid(x) * (1 - sigmoid(x))
    # Da wir hier totalvalue (x) bekommen, berechnen wir sigmoid zuerst:
    s = 1 / (1 + math.exp(-totalvalue)) if totalvalue >= -500 else 0.0
    return s * (1.0 - s)


def elu(totalvalue: float, alpha: float = 1.0) -> float:
    return totalvalue if totalvalue > 0 else alpha * (math.exp(totalvalue) - 1)


def elu_derivative(totalvalue: float, alpha: float = 1.0) -> float:
    return 1.0 if totalvalue > 0 else alpha * math.exp(totalvalue)


def softmax(layerOutputs: list[float]) -> list[float]:
    max_val = max(layerOutputs)
    exp_values = [math.exp(x - max_val) for x in layerOutputs]
    sum_exp = sum(exp_values)
    return [exp / sum_exp for exp in exp_values]


def softmax_derivative(totalvalue: float, *args) -> float:
    return 1.0


class activationType(StrEnum):
    RELU = "ReLU"
    LEAKY_RELU = "LeakyReLU"
    SIGMOID = "Sigmoid"
    ELU = "ELU"
    SOFTMAX = "Softmax"

    def get_activationFunc(self):
        match self:
            case activationType.RELU:
                return ReLU
            case activationType.LEAKY_RELU:
                return LeakyReLU
            case activationType.SIGMOID:
                return sigmoid
            case activationType.ELU:
                return elu
            case activationType.SOFTMAX:
                return None

    def get_activationFunc_derivative(self):
        match self:
            case activationType.RELU:
                return ReLU_derivative
            case activationType.LEAKY_RELU:
                return LeakyReLU_derivative
            case activationType.SIGMOID:
                return sigmoid_derivative
            case activationType.ELU:
                return elu_derivative
            case activationType.SOFTMAX:
                return softmax_derivative
