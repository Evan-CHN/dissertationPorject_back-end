from backend.model.MLModels.CNN.CNNToolKit import as_keras_metric
from backend.model.MLModels.ModelToolKit import ModelToolKit
from keras import Sequential
from keras.layers import Embedding, Dropout, SimpleRNN, Dense
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from pandas import read_csv
from sklearn.model_selection import train_test_split

# loss: 0.2860 - auc: 0.8828 - val_loss: 0.5445 - val_auc: 0.8839
class RNNToolKit(ModelToolKit):
    def __init__(self):
        super().__init__()
        self.model_name = 'RNN'

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
        import tensorflow as tf
        auc_roc = as_keras_metric(tf.metrics.auc)
        model = Sequential()
        model.add(Embedding(output_dim=32,
                            input_dim=1000,
                            input_length=500))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=1))
        model.add(Dense(units=256,
                        activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1,
                        activation='sigmoid'))
        import numpy as np
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc_roc])
        model.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)),
                  epochs=20, batch_size=200)
        self.model = model
        self.model.save('Output/models/' + self.model_name  + '.h5')