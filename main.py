import pandas as pd
from preprocess_text import clean
from Tokenizer import vectorization
from data_pipeline import get_train_val_pipeline
from model import create_model

if __name__ == '__main__':
    df = pd.read_csv('data/tinkoff_otzivi.csv', sep=';',)
    df = df.dropna()
    df['Текст'] = clean(df['Текст'], normalize_form=True)
    vectorize_layer = vectorization(df['Текст'], max_features=5000, max_sequence_length=128)

    train_data, valid_data = get_train_val_pipeline(df['Текст'], df['positive'], test_size = 0.2, shuffle_train=True, shuffle_val=True,
                           val_batch_size=128, train_batch_size=128, shuffle_buffer_size = 10000)

    model = create_model(vectorize_layer, architecture='GRU', n_rnn_layers=3, rnn_units=256, loss=2e-5, embed_output_dim=256, conv1D=True,
                         conv1D_filters=32, bidirectional=False, dropout=0.5)

    model.fit(train_data, validation_data=valid_data, epochs=200, verbose=1)