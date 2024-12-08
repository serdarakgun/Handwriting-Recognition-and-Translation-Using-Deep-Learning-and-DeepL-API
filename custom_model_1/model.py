
from keras._tf_keras.keras import models, layers

def build_crnn(input_shape, vocab_size):
    inputs = layers.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and Recurrent layers
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Fully connected layers
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    return models.Model(inputs, outputs)
