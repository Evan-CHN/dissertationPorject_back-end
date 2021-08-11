import pandas
import tensorflow as tf
from backend.model.MLModels.CNN.CNNToolKit import as_keras_metric
from backend.model.MLModels.ModelToolKit import ModelToolKit
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.models import Sequential
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# loss: 0.1296 - auc: 0.9313 - val_loss: 0.5785 - val_auc: 0.9322
class LSTMToolKit(ModelToolKit):
    def __init__(self):
        super().__init__()

    def train_model(self):
        csv_file = pandas.read_csv('../../dataset/IMDB Dataset.csv').sample(n=5000)
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
            if item=='negative':
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
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(units=256,
                        activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1,
                        activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=[auc_roc])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=200)

        self.model = model
        self.model.save('Output/models/LSTM_' + self.vectorized_method + '_model.h5')
