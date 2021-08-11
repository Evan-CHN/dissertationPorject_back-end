import numpy as np
import pandas
from backend.model.MLModels.CNN.CNNToolKit import as_keras_metric
from flask import Flask
from flask import request
from flask_cors import CORS
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from pandas import read_csv
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    app.run()


@app.route('/get_vecs', methods=['POST'])
def get_vecs():
    vectors_method = request.get_json()['vecs_method']
    decomposition_method = request.get_json()['decomposition_method']
    base_path = 'backend/model/MLModels/Output/points/'
    neg_vecs = np.load(base_path + decomposition_method + '_neg_' + vectors_method + '.npy', allow_pickle=True)
    pos_vecs = np.load(base_path + decomposition_method + '_pos_' + vectors_method + '.npy', allow_pickle=True)
    global NEG_POINTS
    global POS_POINTS
    NEG_POINTS = neg_vecs.tolist()
    POS_POINTS = pos_vecs.tolist()
    return {'positive': pos_vecs.tolist(), 'negative': neg_vecs.tolist()}


# get detail of 'clicked' point and show
@app.route('/get_vec_detail', methods=['POST'])
def get_vec_detail():
    vec_id = request.get_json()['vec_id']
    vec_sentiment = request.get_json()['vec_sentiment']
    decomposition_method = request.get_json()['decomposition_method']
    vectorized_method = request.get_json()['vectorized_method']
    processed_review_df = pandas.read_csv('backend/dataset/Processed_Dataset.csv')
    map_file = pandas.read_csv(
        'backend/dataset/' + vectorized_method + '_' + decomposition_method + '_' + vec_sentiment.lower() + '_index_map.csv')
    original_id = map_file[map_file['current_dataindex'] == vec_id].original_index.tolist()[0]
    original_review_df = pandas.read_csv('backend/dataset/5000_corpus.csv')
    processed_review = processed_review_df[processed_review_df['index'] == original_id].review.tolist()[0]
    original_review = original_review_df[processed_review_df['index'] == original_id].review.tolist()[0]
    return {
        'processed_review': processed_review,
        'original_review': original_review,
        'sentiment': 'positive' if vec_sentiment == 'POS' else 'negative'
    }


# get corpus details
@app.route('/get_corpus_stat', methods=['GET'])
def get_corpus_stat():
    review_length = {
        '<=100': 0,
        '100-500': 0,
        '500-1000': 0,
        '1000-1500': 0,
        '1500-2000': 0,
        '>=2000': 0
    }
    sentiment_stat = {
        'num_pos_reviews': 0,
        'num_neg_reviews': 0
    }
    csv_file = pandas.read_csv('backend/dataset/5000_corpus.csv')
    for index, item in enumerate(csv_file['review']):
        if len(item) <= 100:
            review_length['<=100'] += 1
        elif len(item) <= 500:
            review_length['100-500'] += 1
        elif len(item) <= 1000:
            review_length['500-1000'] += 1
        elif len(item) <= 1500:
            review_length['1000-1500'] += 1
        elif len(item) <= 2000:
            review_length['1500-2000'] += 1
        elif len(item) > 2000:
            review_length['>=2000'] += 1
        if csv_file['sentiment'][index] == 'positive':
            sentiment_stat['num_pos_reviews'] += 1
        else:
            sentiment_stat['num_neg_reviews'] += 1
    return {
        'review_length_data': review_length,
        'sentiment_polarity': sentiment_stat
    }


# user manipulate ANN model
@app.route('/play_ann_model', methods=['POST'])
def get_ann_model():
    import time
    time_start = time.time()
    from keras.models import load_model
    import tensorflow as tf
    auc_roc = as_keras_metric(tf.metrics.auc)
    model = load_model('backend/model/MLModels/Output/models/' + request.get_json()['model_type'] + '.h5',
                       custom_objects={'auc': auc_roc})
    model.compile(optimizer=request.get_json()['optimizer'], loss='binary_crossentropy', metrics=[auc_roc])
    csv_file = read_csv('backend/dataset/IMDB Dataset.csv').sample(n=5000)
    train_review, test_review, train_sentiment, test_sentiment = train_test_split(csv_file['review'],
                                                                                  csv_file['sentiment'],
                                                                                  test_size=0.3)
    tokenizer = Tokenizer(num_words=1000)
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
    history = model.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)),
                        epochs=request.get_json()['epoch'], batch_size=request.get_json()['batch_size'])
    time_end = time.time()
    return {'val_loss': np.array(history.history['val_loss']).tolist(),
            'train_loss': np.array(history.history['loss']).tolist(),
            'val_acc': np.array(history.history['val_auc']).tolist(),
            'train_acc': np.array(history.history['auc']).tolist(),
            'time_used/s': time_end - time_start,
            'epoch':request.get_json()['epoch']}
