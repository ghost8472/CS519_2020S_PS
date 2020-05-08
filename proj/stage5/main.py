#!/usr/bin/env python3
import argparse, csv, os
import numpy as np
import tensorflow as tf
import pandas as pd
from argparse import RawTextHelpFormatter
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
import matplotlib.pyplot as plt

# Taken from Tensorflow documentation
def df_to_dataset(dataframe, shuffle=True, batch_size=32, name='Name'):
  dataframe = dataframe.copy()
  labels = dataframe.pop(name).values.reshape(-1,1)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', dest='csv', type=str, required=True,
            help='specify *relative* location of input data. (Expects CSV)\n')
    parser.add_argument('-b', dest='batch', type=int, required=False,
            default=100,
            help='specify batch size, default is 100\n')
    parser.add_argument('-s', dest='size', type=int, required=False,
            default=10000,
            help='specify buffer size, default is 10,000\n')
    parser.add_argument('-e', dest='epoch', type=int, required=False,
            default=10,
            help='specify the epoch, default is 10\n')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=['AgeuponOutcome', 'AnimalType-Dog',
                                        'Name', 'SexuponOutcome-Female',
                                        'OutcomeType'])

    train, test = train_test_split(df, test_size=0.4)
    train, valid = train_test_split(train, test_size=0.4)

    '''
    le = LabelEncoder().fit(['Return_to_owner', 'Adoption', 'Euthanasia',
                           'Transfer', 'Died'])
    y_train = le.transform(y_train)
    y_test  = le.transform(y_test)
    '''


    train_ds = df_to_dataset(train, batch_size=args.size)
    valid_ds = df_to_dataset(valid, batch_size=args.size)
    test_ds  = df_to_dataset(test, batch_size=args.size)

    outcome = feature_column.categorical_column_with_vocabulary_list(
            'OutcomeType', ['Return_to_owner', 'Adoption', 'Euthanasia',
                           'Transfer', 'Died'])
    feature = feature_column.indicator_column(outcome)

    feature_layer = tf.keras.layers.DenseFeatures([feature], trainable=False)

    model = tf.keras.Sequential([feature_layer,
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(1)
                               ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=args.epoch, validation_data=valid_ds)

    hist = history.history
    x_arr = np.arange(len(hist['loss'])) +1

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1,2,2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    plt.show()

if __name__ == "__main__":
    main()

