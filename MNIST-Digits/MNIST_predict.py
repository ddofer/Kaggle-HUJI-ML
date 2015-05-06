import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics
from helpers import images_to_data, nudge_dataset, rotate_dataset, crop_images
from helpers import compress_images, get_test_data_set, write_predictions_to_csv
from helpers import load_training_data
import numpy as np
import sklearn.decomposition as deco
import pandas as pd
from sklearn import linear_model
# from nolearn.dbn import DBN
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.linalg import pinv2,pinv
# from numpy.linalg import pinv2 as np_pinv2 # NumPy version is different

"Modified from: https://github.com/nicholaslocascio/kaggle-mnist-digits/blob/master/predict.py"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

def eval_model(clf, X, y, cv=3, n_jobs=2):
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
    print("cv score: {:.3f} +/-{:.3f}".format(np.mean(scores), np.std(scores)))

def inplace_relu(X):
    X[X < 0] = 0
    return X


def get_activation(name):
    if name is None:
        return lambda x: x
    elif name == 'tanh':
        return np.tanh
    elif name == 'relu':
        return inplace_relu
    else:
        raise ValueError("unknown activation: " + name)

class AEELMTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=500, activation='tanh',
                 alphas=np.logspace(-5, 5, 11), orthogonalize=True,
                 random_state=None):
        self.n_components = n_components
        self.alphas = alphas
        self.random_state = random_state
        self.activation = activation
        self.orthogonalize = orthogonalize

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        rnd_proj = rng.normal(size=(n_features, self.n_components))
        if self.orthogonalize:
            if self.n_components > n_features:
                raise ValueError('Cannot orthogonalize with so many components')
            rnd_proj, _ = np.linalg.qr(rnd_proj)
        X_proj = get_activation(self.activation)(np.dot(X, rnd_proj))

        self.ridge_ = ridge = RidgeCV(alphas=self.alphas) #,gcv_mode="eigen")

        ridge.fit(X_proj, X)
        self.components_ = ridge.coef_.T
        return self

    def transform(self, X, y=None):
        projected = np.dot(X, self.components_.T)
        return get_activation(self.activation)(projected)



def run():
    X_train, Y_train = load_training_data()
    """
    X_train=crop_images (X_train)
    print("Train images cropped")
    print(X_train.shape())
    """

    X_train, Y_train = rotate_dataset(X_train, Y_train, 1)
    print("Training Data Rotated")
##    X_train, Y_train = nudge_dataset(X_train, Y_train, nudge_size = 2)
##    print("Training Data augmented")

    n_features = X_train.shape[1]
    n_classes = 10

    "Unsupervised learning+pretraining"
    test_data = get_test_data_set()
    #test_data = rotate_dataset(test_data, None, 3)
    # test_data=crop_images (test_data)
    XX = np.vstack((X_train,rotate_dataset(test_data, None, 3)))
    print("Stacked, unlabelled XX.shape: ",XX.shape)

##    aeelm = AEELMTransformer(n_components=700, activation='relu')
##    X_transform = aeelm.fit(XX).transform(XX)
##    print("AE-ELM 1/2")
##
##    aeelm_2 = AEELMTransformer(n_components=640, activation='relu')
##    X_train = aeelm_2.fit(X_transform).transform(aeelm.transform(X_train))
##    print("AE-ELM 2/2 - First autoencoder trained")
    p = Pipeline([
        ('aelm1', AEELMTransformer(n_components=710, activation='tanh')),
        # ('aelm2', AEELMTransformer(n_components=670, activation='relu')),
        ('aelm3', AEELMTransformer(n_components=500, activation='tanh')),
        ('aelm4', AEELMTransformer(n_components=400, activation='relu')),
        ('aelm5', AEELMTransformer(n_components=350, activation='relu')),
##        ('aelm6', AEELMTransformer(n_components=480, activation='tanh'))
    ])

    p.fit(XX)
    print("AE-ELM -  autoencoder trained")
    X_train = p.transform(X_train)
    test_data = p.transform(test_data)

    classifier = LogisticRegression(C=100)
    print("AE-ELM Transformed training Perf:")
    eval_model(classifier, X_train, Y_train, cv=2, n_jobs=3)

    classifier = SVC(kernel="rbf", C=2.8, gamma=.0073,cache_size = 6500,verbose=False)

    # classifier = DBN([n_features, 8000, n_classes],
    #     learn_rates=0.4, learn_rate_decays=0.9 ,epochs=25, verbose=1)

    classifier.fit(X_train, Y_train)

    predictions = classifier.predict(test_data)
    write_predictions_to_csv(predictions)


def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)
