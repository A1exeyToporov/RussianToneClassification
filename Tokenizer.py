from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def vectorization(text, max_features=5000, max_sequence_length=128):

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=max_sequence_length)

    vectorize_layer.adapt(text.values)
    return vectorize_layer