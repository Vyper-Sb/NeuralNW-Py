import math
from enum import StrEnum
from collections.abc import Callable


def clamp_probability(p: float) -> float:
    eps = 1e-15
    return max(min(p, 1 - eps), eps)


def mse_loss(predicted: list[float], target: list[float]) -> float:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    total = 0.0

    for p, y in zip(predicted, target):
        error = p - y
        total += 0.5 * error**2

    return total


def mse_error(predicted: list[float], target: list[float]) -> list[float]:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    return [p - y for p, y in zip(predicted, target)]


def binary_cross_entropy_loss(predicted: list[float], target: list[float]) -> float:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    total = 0.0

    for p, y in zip(predicted, target):
        p = clamp_probability(p)

        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))

    return total / len(predicted)


def binary_cross_entropy_error(predicted: list[float], target: list[float]) -> list[float]:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    return [p - y for p, y in zip(predicted, target)]


def categorical_cross_entropy_loss(predicted: list[float], target: list[float]) -> float:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    total = 0.0

    for p, y in zip(predicted, target):
        p = clamp_probability(p)
        total += -y * math.log(p)

    return total


def categorical_cross_entropy_error(predicted: list[float], target: list[float]) -> list[float]:
    if len(predicted) != len(target): raise ValueError("predicted und target müssen gleich lang sein.")

    return [p - y for p, y in zip(predicted, target)]


def sparse_categorical_cross_entropy_loss(
    predicted: list[float],
    target_index: int,
) -> float:
    if target_index < 0 or target_index >= len(predicted): raise ValueError("target_index liegt außerhalb der Output-Dimension.")

    p = clamp_probability(predicted[target_index])

    return -math.log(p)


def sparse_categorical_cross_entropy_error(
    predicted: list[float],
    target_index: int,
) -> list[float]:
    if target_index < 0 or target_index >= len(predicted): raise ValueError("target_index liegt außerhalb der Output-Dimension.")

    errors = predicted.copy()
    errors[target_index] -= 1

    return errors



class LossType(StrEnum):

    MSE = "MSE"
    BINARY_CROSS_ENTROPY = "BinaryCrossEntropy"
    CATEGORICAL_CROSS_ENTROPY = "CategoricalCrossEntropy"
    SPARSE_CATEGORICAL_CROSS_ENTROPY = "SparseCategoricalCrossEntropy"


    def get_loss_function(self) -> Callable:
        match(self):
            case LossType.MSE:
                return mse_loss
            case LossType.BINARY_CROSS_ENTROPY:
                return binary_cross_entropy_loss
            case LossType.CATEGORICAL_CROSS_ENTROPY:
                return categorical_cross_entropy_loss
            case LossType.SPARSE_CATEGORICAL_CROSS_ENTROPY:
                return sparse_categorical_cross_entropy_loss
    

    def get_error_function(self) -> Callable:
        match(self):
            case LossType.MSE:
                return mse_error
            case LossType.BINARY_CROSS_ENTROPY:
                return binary_cross_entropy_error
            case LossType.CATEGORICAL_CROSS_ENTROPY:
                return categorical_cross_entropy_error
            case LossType.SPARSE_CATEGORICAL_CROSS_ENTROPY:
                return sparse_categorical_cross_entropy_error