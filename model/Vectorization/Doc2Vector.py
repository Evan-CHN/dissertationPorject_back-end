import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def transform_data(data):
    x_train = []
    for i, text in enumerate(data):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggedDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def vectorize(train_data, test_data):
    vocab_data = transform_data(train_data['review'])
    train_vecs = []
    test_vecs = []
    model = Doc2Vec(vector_size=1000, window=15, min_count=40,workers=4)
    model.build_vocab(vocab_data)
    model.train(vocab_data, total_examples=model.corpus_count, epochs=12)
    for item in train_data:
        train_vecs.append(model.infer_vector(item.split()))
    for item in test_data['review']:
        test_vecs.append(model.infer_vector(item.split()))
    from sklearn.preprocessing import scale
    return scale(np.array(train_vecs)), scale(np.array(test_vecs))
