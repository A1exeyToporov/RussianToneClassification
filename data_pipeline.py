from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_train_val_pipeline(X, Y, test_size = 0.2, shuffle_train=True, shuffle_val=True,
                           drop_remainder = True, val_batch_size=128, train_batch_size=128,
                           random_state=None, shuffle_buffer_size = 10000):
    """Returns train and validation data in form of tuple(train_data, valid_data)
    test_size - share of data to split as test/validation data
    shuffles train data at each iteration if shuffle_train = True
    shuffles validation data at each iteration if shuffle_train = True
    train_batch_size - number of training examples for each iteration to propagate
    val_batch_size - number of training examples for each iteration to validate model
    """

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state = random_state)
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    if shuffle_train:
        train_data = train_data.shuffle(10000, reshuffle_each_iteration=True)

    if shuffle_val:
        valid_data = valid_data.shuffle(10000, reshuffle_each_iteration=True)

    train_data = (train_data
                  .batch(train_batch_size, drop_remainder=drop_remainder)
                  .prefetch(tf.data.experimental.AUTOTUNE))

    valid_data = (valid_data
                  .batch(val_batch_size, drop_remainder=drop_remainder)
                  .prefetch(tf.data.experimental.AUTOTUNE))

    return (train_data, valid_data)