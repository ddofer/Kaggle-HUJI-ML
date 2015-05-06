
'http://nbviewer.ipython.org/github/ogrisel/notebooks/blob/master/representations/Autoencoder%20ELMs.ipynb'
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV,LassoCV,LassoLarsCV,ElasticNetCV,MultiTaskElasticNetCV
from sklearn.utils import check_random_state
import numpy as np

from scipy.linalg import pinv2,pinv
# from numpy.linalg import pinv2 as np_pinv2 # NumPy version is different

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.random_projection import sparse_random_matrix



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


' Seperate source/file! '

class AEELMTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=60, activation='tanh',
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

        # ridge = RidgeCV(alphas=self.alphas) #,normalize=True) #DEFAULT.
        # ridge = LassoLarsCV(n_jobs=-1)
        # ridge =MultiTaskElasticNetCV(alphas=self.alphas, n_jobs=-1)

        self.ridge_ = ridge = RidgeCV(alphas=self.alphas)
        ridge.fit(X_proj, X)

        self.components_ = ridge.coef_.T
        return self

    def transform(self, X, y=None):
        projected = np.dot(X, self.components_.T)
        return get_activation(self.activation)(projected)



# Two-layer, sigmoid feedforward network
# trained using the "Extreme Learning Machine" algorithm.
# Adapted from https://gist.github.com/larsmans/2493300

# TODO: make it possible to use alternative linear classifiers instead
# of pinv2, e.g. SGDRegressor
# TODO: implement partial_fit and incremental learning
# TODO: tr



def relu(X):
    """Rectified Linear Unit"""
    return np.clip(X, 0, None)


class ELMClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Extreme Learning Machine Classifier

    https://gist.github.com/ogrisel/8183636 - OGGrisel

    Basically a 1 hidden layer MLP with fixed random weights on the input
    to hidden layer.

    TODO: document parameters and fitted attributes.
    """

    activations = {
        'tanh': np.tanh,
        'relu': relu,
    }

    def __init__(self, n_hidden=1000, rank=None, activation='tanh',
                 random_state=None, density='auto'):
        self.n_hidden = n_hidden
        self.rank = rank
        if activation is not None and activation not in self.activations:
            raise ValueError(
                "Invalid activation=%r, expected one of: '%s' or None"
                % (activation, "', '".join(self.activations.keys())))
        self.activation = activation
        self.density = density
        self.random_state = random_state

    def fit(self, X, y):
        if self.activation is None:
            # Useful to quantify the impact of the non-linearity
            self._activate = lambda x: x
        else:
            self._activate = self.activations[self.activation]
        rng = check_random_state(self.random_state)

        # one-of-K coding for output values
        self.classes_ = unique_labels(y)
        Y = label_binarize(y, self.classes_)

        # set hidden layer parameters randomly
        n_features = X.shape[1]
        if self.rank is None:
            if self.density == 1:
                self.weights_ = rng.randn(n_features, self.n_hidden)
            else:
                self.weights_ = sparse_random_matrix(
                    self.n_hidden, n_features, density=self.density,
                    random_state=rng).T
        else:
            # Low rank weight matrix
            self.weights_u_ = rng.randn(n_features, self.rank)
            self.weights_v_ = rng.randn(self.rank, self.n_hidden)
        self.biases_ = rng.randn(self.n_hidden)

        # map the input data through the hidden layer
        H = self.transform(X)

        # fit the linear model on the hidden layer activation
        self.beta_ = np.dot(pinv2(H), Y)
        return self

    def transform(self, X):
        # compute hidden layer activation
        if hasattr(self, 'weights_u_') and hasattr(self, 'weights_v_'):
            projected = safe_sparse_dot(X, self.weights_u_, dense_output=True)
            projected = safe_sparse_dot(projected, self.weights_v_)
        else:
            projected = safe_sparse_dot(X, self.weights_, dense_output=True)
        return self._activate(projected + self.biases_)

    def decision_function(self, X):
        return np.dot(self.transform(X), self.beta_)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

