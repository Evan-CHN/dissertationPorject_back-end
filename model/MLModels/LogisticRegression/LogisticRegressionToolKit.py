import joblib
import numpy as np
from backend.model.MLModels.ModelToolKit import ModelToolKit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV


class LRToolKit(ModelToolKit):
    def __init__(self):
        super().__init__()
        self.hyper_parameters = {"penalty": ["l1", "l2"], "C": np.logspace(0.01, 10, 10)}
        self.model_name = 'LR'

    def train_model(self):
        model = LogisticRegression(solver='liblinear', max_iter=1e4)
        print('selecting hyperparameters for C and Penalty...')
        grid = GridSearchCV(model, param_grid=self.hyper_parameters, cv=5, scoring=make_scorer(f1_score))
        grid.fit(self.train_review, self.train_sentiment)
        self.model = grid.best_estimator_
        print('model trained!')
        print(grid.best_params_)
        print(grid.best_score_)

    def model_evaluation(self):
        predict_y = self.model.predict(self.test_review)
        mse = np.average((predict_y - self.test_sentiment) ** 2)
        rmse = np.sqrt(mse)
        print("rmse:", rmse)
        acc = 100 * np.mean(predict_y == self.test_sentiment)
        print("acc:", acc)

    def save_model(self):
        joblib.dump(self.model, self.model_name + '_' + self.vectorized_method + '_model.pkl')
