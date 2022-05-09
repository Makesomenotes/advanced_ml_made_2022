import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import sparse as sp

from assignment_hw2_module import get_scores


def log_loss(y_true, y_pred):
    return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class EMClassifier:
    def __init__(self, model, w=None):
        self.w = np.hstack([model.intercept_, model.coef_[0]])
        self.scores = []
        
    def _add_intercept(self, X):
        return sp.hstack((np.ones((X.shape[0], 1)), X), format='csr')
    
    def _init_w(self, dim):
        self.w = np.random.randn(dim)
        
    def _E_step(self, data, preds):
        team_strength = pd.DataFrame({'team_id': data['team_id'],
                                      'question_id': data['question_id'], 
                                      'team_strength': 1 - preds})
        team_strength = team_strength.groupby(['team_id', 'question_id']).agg({'team_strength': 'prod'}).reset_index()
        team_strength['team_strength'] = 1 - team_strength['team_strength']
        team_strength = data[['team_id', 'question_id']].merge(team_strength)
        y = np.clip(preds / team_strength['team_strength'], 0, 1).values # переведем к вероятностям
        y[data['answer'] == 0] = 0
        return y
        
    def _M_step(self, X, y):
        min_loss = np.inf
        indices = np.arange(X.shape[0])
        for _ in range(100):
            indices = np.random.permutation(indices)
            for batch_idx in np.array_split(indices, len(indices) // 1000):
                x_batch, y_batch = X[batch_idx], y[batch_idx]
                grad = x_batch.T.dot(self.predict(x_batch) - y_batch) / len(y_batch)
                self.w -= 1.5 * grad
                
            cur_loss = log_loss(y, self.predict(X))
            if min_loss - cur_loss < 1e-6:
                break
                
            min_loss = cur_loss
                
    def fit(self, X_tr, train_data, X_te=None, test_data=None):
        X_tr = self._add_intercept(X_tr)
        if self.w is None or len(self.w) != X_tr.shape[1]:
            self._init_w(X_tr.shape[1])
        
        for iter_ in range(10): 
            print('iter {}'.format(iter_))
            preds = self.predict(X_tr)
            y = self._E_step(train_data, preds)
            self._M_step(X_tr, y)
            self.scores.append(get_scores(test_data, self.predict(X_te)))
                         
    def predict(self, X):
        if len(self.w) != X.shape[1]:
            X = self._add_intercept(X)
        return expit(X.dot(self.w)) 