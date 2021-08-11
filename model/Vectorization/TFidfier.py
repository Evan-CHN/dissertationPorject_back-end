from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def vectorize(train_data, test_data):
    tf = TfidfVectorizer(max_features=1000,ngram_range=(1, 2))
    x_train = tf.fit_transform(train_data['review']).toarray()
    x_test = tf.transform(test_data['review']).toarray()
    from sklearn.preprocessing import scale
    return scale(x_train), scale(x_test)
