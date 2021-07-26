from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Embedding, Dropout, LeakyReLU, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision


def create_model(vectorize_layer, architecture, n_rnn_layers, rnn_units, loss=2e-5,
                 embed_output_dim=256, conv1D=True, conv1D_filters=32, bidirectional=True, dropout=0.5):

    """ This function creates sequence of layers for RNN model
    vectorize_layer - layer of vectorized text
    architecture - type of RNN layers GRU/LSTM
    n_rnn_layers - number of GRU/LSTM to stack
    rnn_units - rnn_layers dimension
    """
    sequence = [vectorize_layer,
                Embedding(
                    input_dim=len(vectorize_layer.get_vocabulary())+1,
                    output_dim=embed_output_dim,
                    mask_zero=True)
                ]

    if conv1D:
        sequence.append(Conv1D(filters=conv1D_filters, kernel_size=8, strides=1, padding='same'))

    sequence.append(LeakyReLU(alpha=0.4))
    sequence.append(Dropout(dropout))

    if architecture == "GRU":
        if bidirectional:
            for i in range(n_rnn_layers - 1):
                sequence.append(Bidirectional(GRU(rnn_units, return_sequences=True)))
            sequence.append(Bidirectional(GRU(rnn_units)))
        else:
            for i in range(n_rnn_layers - 1):
                sequence.append(GRU(rnn_units, return_sequences=True))
            sequence.append(GRU(rnn_units))

    if architecture == "LSTM":
        if bidirectional:
            for i in range(n_rnn_layers - 1):
                sequence.append(Bidirectional(LSTM(rnn_units, return_sequences=True)))
            sequence.append(Bidirectional(LSTM(rnn_units)))
        else:
            for i in range(n_rnn_layers - 1):
                sequence.append(LSTM(rnn_units, return_sequences=True))
            sequence.append(LSTM(rnn_units))

    sequence.append(Dropout(0.5))
    sequence.append(Dense(256))
    sequence.append(LeakyReLU(alpha=0.4))
    sequence.append(Dropout(0.5))
    sequence.append(Dense(1, activation='sigmoid'))

    model = Sequential(sequence)

    model.compile(loss=BinaryCrossentropy(),
                  optimizer=Adam(2e-5),
                  metrics=[Recall(), Precision(), 'accuracy'])

    return model
