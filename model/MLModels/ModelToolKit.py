import os
import pathlib

import numpy as np
import pandas
from backend.model import PCA, UMAP
from backend.model.TextProcess import text_process
from backend.model.Vectorization import TFidfier, Doc2Vector, Word2Vector
from sklearn.model_selection import train_test_split


# blueprint for all "xxxToolKit" class
class ModelToolKit():
    def __init__(self):
        self.dataset = pandas.read_csv('../../dataset/IMDB Dataset.csv', encoding='UTF-8')
        path = pathlib.Path('../../dataset/Processed_Dataset.csv')
        self.if_text_processed = True if path.exists() else False
        self.if_vectorization = False
        self.if_hyper_parameters_selection = False
        self.model = None
        self.train_review = None
        self.test_review = None
        self.train_sentiment = None
        self.test_sentiment = None
        self.dataset = self.dataset.sample(n=5000)
        self.vectorized_method = ''
        self.PCA_coordinates = {
            'positive_review': None,
            'negative_review': None
        }
        self.UMAP_coordinates = {
            'positive_review': None,
            'negative_review': None
        }

    def text_processed(self):
        if not self.if_text_processed:
            self.dataset.reset_index().to_csv('../../dataset/5000_corpus.csv')
            print('start text processing..')
            self.dataset = text_process(self.dataset)
            print('original text processed!')
            self.dataset.to_csv('../../dataset/Processed_Dataset.csv')
        else:
            self.dataset = pandas.read_csv('../../dataset/Processed_Dataset.csv', encoding='UTF-8')
        self.if_text_processed = True

    # front-end ensure that "vectorized_method" is legal
    def vectorization(self, vectorized_method):
        self.vectorized_method = vectorized_method
        train_data, test_data = train_test_split(self.dataset, test_size=0.3)
        # data = train_data['Unnamed: 0']
        if self.if_text_processed:
            if not os.path.isfile('Output/vectors/' + vectorized_method + '_train_vecs.npy') and not os.path.isfile(
                    'Output/vectors/' + vectorized_method + '_test_vecs.npy'):
                print('using ' + vectorized_method + ' method to vectorize the dataset...')
                self.train_sentiment = train_data['sentiment']
                self.test_sentiment = test_data['sentiment']
                if vectorized_method == 'TF-IDF':
                    self.train_review, self.test_review = TFidfier.vectorize(train_data, test_data)
                elif vectorized_method == 'Doc2Vec':
                    self.train_review, self.test_review = Doc2Vector.vectorize(train_data, test_data)
                elif vectorized_method == 'Word2Vec':
                    self.train_review, self.test_review = Word2Vector.vectorize(train_data, test_data)
                np.save('Output/vectors/' + vectorized_method + '_train_vecs.npy', self.train_review)
                np.save('Output/vectors/' + vectorized_method + '_test_vecs.npy', self.test_review)
                np.save('Output/vectors/' + vectorized_method + '_train_sentiment.npy', self.train_sentiment)
                np.save('Output/vectors/' + vectorized_method + '_test_sentiment.npy', self.test_sentiment)
            else:
                self.train_review = np.load('Output/vectors/' + vectorized_method + '_train_vecs.npy',
                                            allow_pickle=True)
                self.test_review = np.load('Output/vectors/' + vectorized_method + '_test_vecs.npy', allow_pickle=True)
                self.train_sentiment = np.load('Output/vectors/' + vectorized_method + '_train_sentiment.npy',
                                               allow_pickle=True)
                self.test_sentiment = np.load('Output/vectors/' + vectorized_method + '_test_sentiment.npy',
                                              allow_pickle=True)

        else:
            print('dataset has not been processed!')
        self.if_vectorization = True
        print("saving data...")
        review_data = np.concatenate((self.train_review, self.test_review))
        index_data = np.concatenate((train_data['index'], test_data['index']))
        sentiment_data = np.concatenate((self.train_sentiment, self.test_sentiment))
        PCA.PCAdecomposition(
            review_data, sentiment_data, index_data, vectorized_method)
        UMAP.UMAPdecomposition(
            review_data, sentiment_data, index_data, vectorized_method)
