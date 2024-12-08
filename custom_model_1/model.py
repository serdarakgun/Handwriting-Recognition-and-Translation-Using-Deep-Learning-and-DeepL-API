from keras._tf_keras.keras import models, layers


def build_crnn(input_shape, vocab_size):
    """
    Builds a CRNN (Convolutional Recurrent Neural Network) model for sequence prediction tasks.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        vocab_size (int): Number of unique output classes (size of the vocabulary).

    Returns:
        keras.Model: Compiled CRNN model.
    """

    # Define the input layer with the given shape
    inputs = layers.Input(shape=input_shape)

    # Convolutional layers for feature extraction
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    # 2D convolutional layer with 32 filters, a kernel size of 3x3, ReLU activation, and same padding.
    x = layers.MaxPooling2D((2, 2))(x)
    # Max pooling layer to reduce spatial dimensions (down-sampling by a factor of 2).
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    # Another convolutional layer with 64 filters for deeper feature extraction.
    x = layers.MaxPooling2D((2, 2))(x)
    # Additional max pooling layer for further reduction in spatial dimensions.

    # Reshape the output of the convolutional layers to match the input format required by the RNN layers
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    # Reshapes the tensor from (height, width, channels) to (time_steps, features_per_step).
    # This prepares the data for the recurrent layer, treating width as time steps.

    # Recurrent layer to model sequential dependencies
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # A bidirectional LSTM layer with 128 units. Processes the sequence from both directions for better context.
    # `return_sequences=True` ensures that the output is a sequence, suitable for further processing.

    # Fully connected layer to produce predictions
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    # A dense layer with `vocab_size` units. Each unit corresponds to a class in the vocabulary.
    # Softmax activation ensures the output is a probability distribution.

    # Create and return the model
    return models.Model(inputs, outputs)
