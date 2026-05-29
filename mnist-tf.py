import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds

# 1. Daten laden
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)  # type: ignore


# 2. Normalisierungs-Funktion
def normalize_img(image, label):
    """Normalisiert Bilder: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# 3. Pipelines verarbeiten
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# 4. Modell mit Softmax in der letzten Schicht
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),  # Softmax ist jetzt aktiv
    ]
)

# 5. Modell kompilieren (from_logits=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),  # Geändert auf False
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# 6. Modell trainieren
model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
