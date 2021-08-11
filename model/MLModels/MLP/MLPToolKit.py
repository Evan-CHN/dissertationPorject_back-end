import tensorflow as tf
from backend.model.MLModels.CNN.CNNToolKit import as_keras_metric
from backend.model.MLModels.ModelToolKit import ModelToolKit
from keras import Sequential
from keras.layers import Dense, Embedding, Flatten

from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from pandas import read_csv
from sklearn.model_selection import train_test_split


# loss: 0.0724 - auc: 0.9072 - val_loss: 0.5247 - val_auc: 0.9097
class MLPToolKit(ModelToolKit):
    def __init__(self):
        super().__init__()
        self.model_name = 'MLP'

    def train_model(self):
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
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        import numpy as np
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc_roc])
        model.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)),
                  epochs=20, batch_size=200)
        self.model = model
        self.model.save('Output/models/' + self.model_name + '.h5')
