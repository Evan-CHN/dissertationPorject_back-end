from gensim.models import Word2Vec
import numpy as np


def averageVector(many_vectors, column_num):
    average_vector = []
    for i in range(0, column_num, 1):
        average_vector.append(0)
    row_num = len(many_vectors)
    row_index = 0
    for weight_index, vector in enumerate(many_vectors):
        for i in range(0, column_num, 1):
            average_vector[i] += float(vector[i])
        row_index += 1
    for i in range(0, column_num, 1):
        average_vector[i] = average_vector[i] / row_num
    return average_vector


def get_sentence_matrix(splited_words, model):
    sentences_matrix = []
    index = 0
    while index < len(splited_words):
        words_matrix = []
        words = splited_words[index].split(" ")
        for word in words:
            if word in model:
                words_matrix.append(np.array(model[word]))
        feature = averageVector(many_vectors=words_matrix,
                                column_num=model.vector_size)
        sentences_matrix.append(feature)
        index += 1
    return sentences_matrix


def vectorize(train_data, test_data):
    train = []
    test = []
    train_vecs = []
    test_vecs = []
    for index, item in enumerate(train_data['review']):
        train.append(item.split(' '))
    for index, item in enumerate(test_data['review']):
        test.append(item.split(' '))
    vocab = train + test
    model = Word2Vec(vocab, vector_size=1000, min_count=1,window=15)
    for sentence in train:
        temp = 1000*[0.0]
        temp = np.array(temp)
        index = 0
        for word in sentence:
            temp+=model.wv[word]
            index += 1
        train_vecs.append(temp / index)
    for sentence in test:
        temp = 1000 * [0.0]
        temp = np.array(temp)
        index = 0
        for word in sentence:
            temp+=model.wv[word]
            index += 1
        test_vecs.append(temp / index)
    from sklearn.preprocessing import scale
    return scale(train_vecs),scale(test_vecs)