import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers


def load_datasets():
    global training, y_train
    training = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
    x_test_orig = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
    training = training.sample(frac=1).reset_index(drop=True)
    y_train = training['label']
    x_train_orig = training.drop('label', axis=1)
    print(y_train.head())
    return x_train_orig, y_train, x_test_orig


def explore_input_shape():
    first_image = x_train_orig.iloc[0]
    print('before reshape:')
    print(f'first_image.shape: {first_image.shape}')
    print(f'type(first_image): {type(first_image)}')
    first_image = first_image.values.reshape(28, 28)
    print('after reshape:')
    print(f'first_image.shape: {first_image.shape}')
    print(f'type(first_image): {type(first_image)}')
    print(f'x_train.shape:     {x_train_orig.shape}')
    reshaped = x_train_orig.apply(lambda row: row.values.reshape(28, 28), axis=1)
    print(f'reshaped.shape:    {reshaped.shape}')


# reshape each image as numpy matrix
def reshape_as_3d_matrix(df):
    tmp_series = df.apply(lambda row: row.values.reshape(28, 28), axis=1)
    return np.reshape(np.concatenate(tmp_series.values), (df.shape[0], 28, 28, 1))


def reshape_and_normalize_datasets():
    x_train, x_test = x_train_orig / 255.0, x_test_orig / 255.0
    print(f'x_train.shape: {x_train.shape}')
    print('reshaping...')
    x_train = reshape_as_3d_matrix(x_train)
    x_test = reshape_as_3d_matrix(x_test)
    print(f'x_train.shape: {x_train.shape}')
    return x_train, x_test


def explore_training_set():
    for i in range(10):
        plt.figure()
        sns.heatmap(x_train[i, :, :, 0])
        plt.title(y_train.iloc[i])
        plt.show()


def build_and_compile_model(num_of_filters=30, filter_size=5):
    # Build the model
    model = keras.models.Sequential([
        keras.layers.Conv2D(num_of_filters, (filter_size, filter_size), activation='relu', input_shape=(28, 28, 1),
                            padding='same',
                            kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(num_of_filters, (filter_size, filter_size), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='relu')
    ])
    print(model.summary())
    # Construct loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # compile loss function into model
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train):
    # train model to minize loss
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=1)


def train_and_validate_model(model, x_train, y_train, early_stopping_callback, validation_fraction=0.1):
    split_index = int((1 - validation_fraction) * x_train.shape[0])
    x_do_train = x_train[0:split_index]
    x_valid = x_train[split_index:x_train.shape[0]]

    y_do_train = y_train[0:split_index]
    y_valid = y_train[split_index:]
    print(f'training_set size: {x_do_train.shape[0]}')
    print(f'validation_set size: {x_valid.shape[0]}')
    # train model to minize loss
    model.fit(
        x_do_train,
        y_do_train,
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping_callback],
        validation_data=(x_valid, y_valid),
        verbose=1)


def predict(model, x_test):
    prediction_scores = model.predict(x_test)
    print(f'prediction_scores shape: {prediction_scores.shape}')
    probabilities = pd.DataFrame(tf.nn.softmax(prediction_scores))
    print(type(probabilities))
    print(f'probabilities shape: {probabilities.shape}')
    return probabilities


def convert_model_outputs_to_decisions(probabilities):
    predictions = pd.DataFrame(probabilities.apply(np.argmax, axis=1), columns=['Label'])
    print(f'predictions shape: {predictions.shape}')
    print(f'predictions type: {type(predictions)}')
    predictions.index += 1
    predictions.head()
    return predictions


x_train_orig, y_train, x_test_orig = load_datasets()
explore_input_shape()
x_train, x_test = reshape_and_normalize_datasets()
explore_training_set()
model = build_and_compile_model()
# stop when model stops to improve on validation set
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.1, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)

train_and_validate_model(model, x_train, y_train, early_stopping_callback)
train_model(model, x_train, y_train)
probabilities = predict(model, x_test)
predictions = convert_model_outputs_to_decisions(probabilities)
predictions.to_csv('predictions_cnn_2layers_32dense_l2regression.csv', index_label='ImageId')