import functools

import tensorflow as tf
from backend.model.MLModels.ModelToolKit import ModelToolKit
from keras import backend as K
from keras.layers import Dropout, Dense, Conv1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from pandas import read_csv
from sklearn.model_selection import train_test_split


# loss: 0.0262 - auc: 0.9392 - val_loss: 0.5243 - val_auc: 0.9408

def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


class CNNToolKit(ModelToolKit):
    def __init__(self):
        super().__init__()
        self.model_name = 'CNN'

    def train_model(self):
        import numpy as np
        self.train_review = np.array(self.train_review)
        self.train_sentiment = np.array(self.train_sentiment)
        self.test_review = np.array(self.test_review)
        self.test_sentiment = np.array(self.test_sentiment)
        csv_file = read_csv('../../dataset/IMDB Dataset.csv').sample(n=5000)
        max_features = 1000
        train_review, test_review, train_sentiment, test_sentiment = train_test_split(csv_file['review'],
                                                                                      csv_file['sentiment'],
                                                                                      test_size=0.3)
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(train_review)
        x_train_seq = tokenizer.texts_to_sequences(train_review)
        x_test_seq = tokenizer.texts_to_sequences(test_review)
        x_train = sequence.pad_sequences(x_train_seq, maxlen=500)
        x_test = sequence.pad_sequences(x_test_seq, maxlen=500)
        y_train = []
        y_test = []
        for item in train_sentiment:
            if item == 'negative':
                y_train.append(0)
            else:
                y_train.append(1)
        for item in test_sentiment:
            if item == 'negative':
                y_test.append(0)
            else:
                y_test.append(1)
        auc_roc = as_keras_metric(tf.metrics.auc)
        model = Sequential()
        model.add(Embedding(output_dim=32,
                            input_dim=1000,
                            input_length=500))
        model.add(Conv1D(128, 3, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[auc_roc])

        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test), epochs=20,
                  batch_size=200)
        self.model = model
        self.model.save('Output/models/CNN_' + '_model.h5')
