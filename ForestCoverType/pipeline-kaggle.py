import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import Pipeline,make_pipeline,FeatureUnion
from sklearn.decomposition import PCA #, LDA
from sklearn.lda import LDA
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split
# from nolearn.dbn import DBN
from sklearn.cross_validation import StratifiedKFold,cross_val_score,StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.feature_selection import SelectKBest
# from breze.learn.autoencoder import DenoisingAutoEncoder, ContractiveAutoEncoder
# from breze.learn.sparsefiltering import SparseFiltering #as br_SpFilt
# from breze.learn.lde import LinearDenoiser
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import sparse_filtering #.SparseFiltering

from sklearn.base import BaseEstimator, ClassifierMixin
from ELM import ELMClassifier,AEELMTransformer

'Theanets - unsupervised pretraining https://github.com/lmjohns3/theano-nets/issues/39'

def getCVScore(predictor,data,labels,CV=3):
    scores = cross_validation.cross_val_score(predictor,
        X=data, y=labels,cv=StratifiedShuffleSplit(y,n_iter=CV,train_size=0.15),n_jobs=-1)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    'http://stackoverflow.com/questions/21506128/best-way-to-combine-probabilistic-classifiers-in-scikit-learn?rq=1'
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return np.mean(self.predictions_, axis=0)

# class SelfTrainingClassifier(BaseEstimator):
#     '''
#     http://nbviewer.ipython.org/github/ogrisel/notebooks/blob/master/Semi-supervised%20Extra%20Trees.ipynb
#     I modify this to classify with  each iteration the test samples that have
#     the highest probability, and use them to expand""bootstrap" my train
#     data in each subsequent iteration. (Which will also lower the threshhold).
#     '''

#     def __init__(self, base_estimator=None, n_iter=11, clamp_true_target=False):
#         self.base_estimator = base_estimator
#         self.n_iter = n_iter
#         self.clamp_true_target = clamp_true_target

#     def fit(self, X, y, X_unlabeled=None, X_val=None, y_val=None):
#         if self.base_estimator is None:
#             model = ExtraTreesClassifier(n_estimators=200)
#         else:
#             model = clone(self.base_estimator)

#         X_train, y_train = X, y

#         for i in range(self.n_iter):
#             model.fit(X_train, y_train)

#             if X_val is not None and y_val is not None:
#                 print(model.score(X_val, y_val))

#             if self.clamp_true_target:
#                 y_predicted = y
#             else:
#                 y_predicted = model.predict(X)

#             X_train = np.vstack([X, X_unlabeled])
#             y_train = np.concatenate([y, model.predict(X_unlabeled)])

#         self.estimator_ = model

#     def predict(self, X):
#         return self.estimator_.predict(X)

#     def score(self, X, y):
#         return self.estimator_.score(X, y)



if __name__ == "__main__":

    loc_submission = "./kaggle_forest/kaggle.forest.submission.csv"
    loc_train = r"./data/train.csv"
    loc_pred = r"./data/test.csv"
    loc_unlabel=loc_pred
    df_unlabeled=pd.read_csv(loc_unlabel)

    df_train = pd.read_csv(loc_train)
    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

    X = df_train[feature_cols].values #.astype(np.float32)
    X_unlabeled = df_unlabeled[feature_cols].values #.astype(np.float32)

    print(X_unlabeled.shape,X.shape)

    y = df_train['Cover_Type'].values #.astype(int)
    y = np.asarray( y )

    '''
    df_test = pd.read_csv(loc_pred)
    X_test = df_test[feature_cols]
    '''
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.85, random_state=7)

    # 'Feature dimensionality ? :'
    # dim = len(feature_cols)

    SS = StandardScaler()
    # SS.fit(X_unlabeled)

    X_unlabeled = SS.fit_transform(X_unlabeled)
    X=SS.transform(X)
    '''
    MM=MinMaxScaler()
    X_unlabeled = MM.fit_transform(X_unlabeled)
    X=MM.transform(X)
    '''
    # X=SS.fit_transform(X)
    # X=MM.fit_transform(X)

    # X_test=X_unlabeled


    '''
    etrees = ExtraTreesClassifier(n_estimators=950, n_jobs=-1)
    print("Tree cv score:")
    getCVScore(etrees,X,y,CV=3)
    '''

    # labProp_1 = LabelPropagation(kernel="knn")
    # print("labProp_1 cv score:")
    # getCVScore(labProp_1,X,y)


    AE1 = AEELMTransformer(n_components=60, activation='tanh')
    AE2 = AEELMTransformer(n_components=60, activation='tanh')
    AE3 = AEELMTransformer(n_components=60, activation='relu')
    AE4 = AEELMTransformer(n_components=50, activation='tanh')

    # pca = PCA()
    # p1 = make_pipeline(AE2,AE1)

    p2 = make_pipeline(AE3)
    print("p2")


    # # AE1.fit(X)
    # # X_AE1 = AE1.fit(X).transform(X)
    # X_AE1 = AE1.fit(X_unlabeled).transform(X)

    # print(X.shape)
    # print(X_AE1.shape)

    # X_AE2=AE2.fit(X_AE1).transform(X_AE1)

    # print(X_unlabeled.shape,X.shape)
    # AE3.fit(X_unlabeled)
    # # AE3.fit(X)
    # X_AE3=AE3.transform(X)

    '''
    print("X_AE1 cv score:")
    getCVScore(etrees,X_AE1,y)
    print("X_AE1->AE2 cv score:")
    getCVScore(etrees,X_AE2,y)
    print("X-Unlabelled_AE3 cv score:")
    getCVScore(etrees,X_AE3,y)
    '''

    # combined_features = FeatureUnion([("pca", PCA()),("AE1",AE1),("AE3",AE3),("KBest",SelectKBest(k=55))],n_jobs=-1)
    combined_features = FeatureUnion([("p2", p2),("KBest",SelectKBest(k=63))],n_jobs=1)
    # X_4 = combined_features.fit_transform(X,y)

    combined_features.fit(X_unlabeled)
    X_unlabeled = combined_features.transform(X_unlabeled)
    print("test X_unlabelled transformed", ...)
    X_4 = combined_features.transform(X,y)


    '''
    for n_hidden in [ 400, 800, 1500, 4000]:
        print("Fitting relu ELM ; n_hidden=%d..." % n_hidden)
        # tic = time()
        model = ELMClassifier(n_hidden=n_hidden, rank=None, density='auto',
                              activation='relu')

        print("ELM on X score:")
        getCVScore(model,X,y)
        # print("ELM on X + AE-ELM-Unlabelled score:")
        # getCVScore(model,X_AE3,y)
        print("ELM on X AE-stacked score:")
        getCVScore(model,X_4,y)

        print("Trees on X AE-stacked :")
        getCVScore(etrees,X_4,y)

        print("Done")

    for n_hidden in [400,800, 1500, 4000]:
        print("Fitting tanh ELM; n_hidden=%d..." % n_hidden)
        # tic = time()
        model = ELMClassifier(n_hidden=n_hidden, rank=None, density='auto',
                              activation='tanh')

        print("ELM on X score:", ...)
        getCVScore(model,X,y)
        # print("ELM on X + AE-ELM-Unlabelled score:")
        # getCVScore(model,X_AE3,y)
        print("ELM on X AE-ELM-stacked score:")
        getCVScore(model,X_4,y)
        print("Done")
    '''

    clf = ExtraTreesClassifier(n_estimators=430, n_jobs=-1)
    print("Tree cv score on Expanded features:")
    getCVScore(clf,X_4,y,CV=4)
    clf.fit(X_4,y)
    print(X_4.shape)


    # df_test = pd.read_csv(loc_pred)
    # X_test = df_test[feature_cols]

    # X_test = combined_features.transform(X_test)
    'https://gist.github.com/roycoding/4d4951f41904f0f62a6e'

    # test_ids = df_unlabeled['Id']

    # with open(loc_submission, "wt") as outfile:
    #     outfile.write("Id,Cover_Type\n")
    #     # for e, val in enumerate(list(clf.predict(X_test))):
    #     for e, val in enumerate(list(clf.predict(X_4))):
    #         outfile.write(test_ids[e]+","+val+" \n")
    #         # outfile.write("%s,%s\n"%(test_ids[e],val))

    print("predicting unlabelled")
    y_test_rf = clf.predict(X_unlabeled)
    pd.DataFrame({'Id':df_unlabeled.Id.values,'Cover_Type':y_test_rf})\
                .sort_index(ascending=False,axis=1).to_csv("./kaggle_forest/kaggle.forest.submission.csv",index=False)