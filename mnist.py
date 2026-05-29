import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds
import numpy as np
import math
from FSNeuralNetwork.neural_network import *
from FSNeuralNetwork.activation_functions import activationType
from FSNeuralNetwork.startweight_utils import generate_random_weights
from FSNeuralNetwork.loss_functions import LossType

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)  # type: ignore


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

weights_h1 = generate_random_weights(128, 784)
weights_out = generate_random_weights(10, 128)

net = NeuralNetwork(
    inputs=784,
    outputs=10,
    output_bias=0,
    output_activation_type=activationType.SOFTMAX,
    loss_type=LossType.CATEGORICAL_CROSS_ENTROPY,
)

net.add_hidden_layer(
    neurons=128, bias=0, activation_type=activationType.RELU, weights=weights_h1
)
net.outputLayer.set_weights(weights=weights_out)


EPOCHS = 1
LEARNING_RATE = 0.01

print("Starte Training ...")

for epoch in range(EPOCHS):
    print("Epoch ", epoch, "/", EPOCHS)
    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0
    curr_val = 0

    for images_batch, labels_batch in ds_train:
        print("batch ", curr_val, "/", len(ds_train))
        np_images = images_batch.numpy().reshape(-1, 784)
        np_labels = labels_batch.numpy()
        curr_val += 1

        one_hot_labels = np.eye(10)[np_labels]

        avg_loss, _ = net.train_with_sgd(
            np_images.tolist(), one_hot_labels.tolist(), 0.01
        )

        print(avg_loss)

    # Testdaten auswerten
    test_correct = 0
    test_samples = 0

    for test_images_batch, test_labels_batch in ds_test:
        np_test_images = test_images_batch.numpy().reshape(-1, 784)
        np_test_labels = test_labels_batch.numpy()

        for idx in range(len(np_test_images)):
            input_data = np_test_images[idx].tolist()
            target_label = int(np_test_labels[idx])

            predicted_output = net.calculate_data(input_data)
            predicted_label = predicted_output.index(max(predicted_output))

            if predicted_label == target_label:
                test_correct += 1
            test_samples += 1

    # Metriken für die Epoche berechnen
    val_accuracy = test_correct / test_samples

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"test - accuracy: {val_accuracy:.4f}")

print("Training beendet!")
net.save_neural_network("modell6Epochs.json")
