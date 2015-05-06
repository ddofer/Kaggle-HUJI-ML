from sklearn import ensemble
import time
from time import time
from sklearn import metrics, cross_validation
from sklearn import svm, preprocessing
import pandas as pd
from AdaHERF import AdaHERF
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.externals import joblib
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import os
import random
from sklearn.lda import LDA
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA, RandomizedPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from elm import SimpleELMClassifier, ELMClassifier, SimpleELMRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from operator import itemgetter
# from SdA import SdA

SEED = 4  # always use a seed for randomized procedures
N_TRIALS = 4

def CV_multi_stats(X, y, model) :
    '''
    http://scikit-learn.org/stable/modules/model_evaluation.html#classification
    -metrics
    '''
    n = N_TRIALS  # repeat the CV procedure 10 times to get more precise results
    scores = cross_validation.cross_val_score(estimator=model, X=X, y=y, cv=n) #Accuracy
    # scores_f1 = cross_validation.cross_val_score(estimator=model, X=X, y=y,                                                 scoring='f1', cv=n)
    print("Model Accuracy: %0.2f (+- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("Model f1: %0.2f (+- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2))

def CV_Binary_stats(X, y, model) :
    '''
    http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    Note that some of the metrics here ONLY work for BINARY tasks.
    '''
    # global mean_auc, mean_precision, mean_recall, mean_accuracy
    n = N_TRIALS  # repeat the CV procedure 10 times to get more precise results
    mean_accuracy=0
    for i in range(n) :
        # for each iteration, randomly hold out 30% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y,
                                                                         test_size=.3,
                                                                         random_state=i * SEED)
        # train model and make predictions
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        # fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        print("( %d/%d)" % (i + 1, n))
        accuracy = accuracy_score(y_cv, preds)
        mean_accuracy += accuracy
    mean_accuracy = (mean_accuracy / n)
    print('mean_accuracy:  %s ' %(round(mean_accuracy, 2)))

def report(grid_scores, n_top=3) :
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores) :
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.2f} (std: {1:.2f})".format(
            score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# ,
AdaBoost_param_dist = {'base_estimator':[DecisionTreeClassifier(),
                                         RandomForestClassifier(
                                             max_features=0.6,min_samples_leaf=
                                                            2, max_depth= 11,
                                                            min_samples_split=
                                                            1, n_estimators= 40),
     RandomForestClassifier(min_samples_leaf=2,max_depth= 8,
                                             min_samples_split=2)
],'n_estimators':sp_randint(10,27),'algorithm':[
    'SAMME.R'] }

Tree_param_dist = {"max_depth" : sp_randint(7,22),
                   "min_samples_split" : sp_randint(1, 4), "min_samples_leaf" :
    sp_randint(1, 3), "criterion" : ["entropy","gini"],"n_estimators" : [150]}

SVM_param_grid = {'C' : [1, 10, 100, 1000], 'kernel' : ['linear', 'rbf'],
                   'gamma' : [0.001, 0.0001]}
SVM_param_dist = {'C' : stats.expon(scale=100), 'gamma' : stats.expon(scale=.1),
                   'kernel' : ['rbf', 'linear'], 'class_weight' : ['auto']}

SGD_param_dist =    {'n_iter' : sp_randint(4,20),'loss':['hinge','modified_huber'],
      'learning_rate' :['optimal'],'penalty' : ['l1', 'elasticnet'],'l1_ratio' : [
    0.15, 0.3, 0.6],'shuffle':[True]}

KN_param_dist = {'n_neighbors':sp_randint(3,15), 'weights':['distance'],
                 'algorithm':['kd_tree'],
                         'leaf_size':[30], 'p':[1,2], 'metric':['minkowski'] }

# scipy.stats.uniform -  distribution is  between loc and loc + scale.
GBT_param_dist = {'loss':['deviance'], 'learning_rate':uniform(0.05,0.1),
                 'subsample':uniform(0.7,0.3),'min_samples_split':[2,4],
                 'min_samples_leaf':[1,2], 'max_depth':sp_randint(3,10),
                 'max_features':['auto',None]}

# GBT_param_dist = {'loss':['deviance'],'min_samples_split':[2,4],
#                  'min_samples_leaf':[1,2], 'max_depth':sp_randint(3,10),
#                  'max_features':[0.8,'auto',None]}

def GridParamSearch(param_dist, clf, X, y, n_iter_search=8) :
    '''
    Searches using rand.search for best model paramters
    diff paramters searched by model type..
    http://nbviewer.ipython.org/github/treycausey/thespread/blob/master/notebooks
    /basic_random_forest_wp_model.ipynb?create=1
    @param clf: estimator/predictor used.
    @param param_dist: Grid of Parameter ranges to tune for the predictor,
    using randomized CV search.
    '''
    print("Starting grid parameter search")
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, n_jobs=-2)
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)


def GetClassifiersCV(X, y, listclassifiers) :
    for model in listclassifiers :
        print(model)
        model = model()
        CV_multi_stats(X, y, model)


if __name__ == "__main__" :
    # loc_train = r".\kaggle_forest\train.csv"
    loc_train = r".\data\train.csv"
    loc_test = r".\data\test.csv"
    loc_submission = ".\kaggle_forest\kaggle.forest.submission.csv"

    df_train = pd.read_csv(loc_train)
    # df_train = df_train.astype('int32', copy=False)

    feature_cols = [col for col in df_train.columns if
                    col not in ['Cover_Type', 'Id']]

    X = df_train[feature_cols].values
    y = df_train['Cover_Type'].values

#Normalize/scale

    MM_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    Scaler = StandardScaler(with_mean=True, with_std=False, copy=True)
    X = Scaler.fit_transform(X.astype(np.float))
    X = MM_scaler.fit_transform(X.astype(np.float))


    # listclassifiers = [AdaBoostClassifier, ELMClassifier, GaussianNB,
    #                    SimpleELMClassifier,
    #                    KNeighborsClassifier, LinearSVC, GradientBoostingClassifier,
    #                    SGDClassifier, RandomForestClassifier,
    #                    SVC ]
    listclassifiers = [MultinomialNB, BernoulliNB,
                       SimpleELMClassifier,
                       KNeighborsClassifier, LinearSVC, GradientBoostingClassifier,
                        RandomForestClassifier]
    # , AdaHERF
    listC2 = [GaussianNB,SimpleELMClassifier,
                       KNeighborsClassifier, LinearSVC, GradientBoostingClassifier,
                        RandomForestClassifier]
    '''
    # l = [m() for m in listclassifiers]
    # print (l)
    '''

    print("models perf:")
    'GetClassifiersCV(X, y, listclassifiers) '

    print("\n SCALED models: \n")
    # SVD = TruncatedSVD(n_components=47, algorithm='randomized', n_iter=5)
    print(X.shape)

    red_PCA = PCA(copy=True, whiten=False)
    # red_PCA.fit(X)
    # X_PCA = red_PCA.transform(X) #Buggy with negative values for NB!
    MM_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    # X_PCA = MM_scaler.fit_transform(X_PCA)

    # print("LDA models:")
    # LDA = LDA(n_components=6)
    # # X_LDA = LDA.fit_transform(X,y)
    # X_LDA = LDA.fit_transform(X_PCA, y)

    # AB = AdaBoostClassifier()
    # print("classic adaboost + PCA - best perf")
    # GridParamSearch(AdaBoost_param_dist, AB, X_PCA, y)

    # print("classic adaboost best perf")
    # GridParamSearch(AdaBoost_param_dist, AB, X, y)

    # print ("AdaHerf performance: \n X")
    # adaH = AdaHERF()
    # CV_Binary_stats(X, y, adaH)
    # print("AdaHerf performance: \n X_PCA ")
    # CV_Binary_stats(X_PCA, y, adaH)
    # print("ADA done")

    # GetClassifiersCV(X_LDA, y, listC2)
    # print("LDA Done")
    # print("PCA models perf:")
    # GetClassifiersCV(X_PCA, y, listC2)
    # print("PCA perf. Done")
    #


    # print("RandCV Best params: RandForest")
    # RF = RandomForestClassifier()
    # GridParamSearch(Tree_param_dist, RF, X_PCA, y)

    # sda = SdA(
    #         numpy_rng=numpy_rng,
    #         n_ins=56,
    #         hidden_layers_sizes=[150, 80, 40],
    #         n_outs=7
    #     )

    ##################################
    'RBM feature learning + CV'
    'http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/'
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import BernoulliRBM
    from sklearn.grid_search import GridSearchCV
    from sklearn.pipeline import Pipeline

    # initialize the RBM + Logistic Regression pipeline
    rbm = BernoulliRBM()
    rbm2 = BernoulliRBM(learning_rate=0.1,n_components=280,n_iter = 250,batch_size=90)
    logistic = LogisticRegression()
    RF = RandomForestClassifier(n_estimators= 380,
                                max_features=0.3)

    # classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier = Pipeline([("rbm", rbm),("RF", RF)])

    # perform a grid search on the learning rate, number of
    # iterations, and number of components on the RBM and
    # C for Logistic Regression
    print( "SEARCHING RBM + Classifier")
    # params = {"rbm__learning_rate": [0.1, 0.01, 0.001], "rbm__n_iter" : [20, 50],
    #           "rbm__n_components" : [256,500], "logistic__C" : [1.0, 10.0]}
    params = {"rbm__learning_rate": [0.1,0.01,1], "rbm__n_iter" : [50,800],
              "rbm__n_components" : [30,65,300],'rbm__batch_size':[50],
              'RF__criterion':  ['entropy']} #'gini']} #'entropy',


    # perform a grid search over the parameter
    start = time()
    gs = GridSearchCV(classifier, params, n_jobs=-2, verbose=1)
    # gs.fit(trainX, trainY)
#    X_RBM = rbm2.fit_transform(X,y)
#    print("X_RBM:",X_RBM.shape)
#    gs.fit(X_RBM, y)

    gs.fit(X, y)

    # print diagnostic informationand grab the best model
    print(    "best score: %0.3f" % (gs.best_score_))
    print(    "RBM + classifier PARAMETERS")
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()) :
#        print( "\t %s: %f" % (p, bestParams[p]))
        print( "\t",p,bestParams[p])

#    print(X.shape)

