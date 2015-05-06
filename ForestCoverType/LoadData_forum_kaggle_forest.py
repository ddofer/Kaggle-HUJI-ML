import pandas as pd
from sklearn import ensemble
# from linear_msda import mSDA
# from SdA import SdA

# from DBN import DBN
# import dA
import numpy as np
# from NeuralNetworksToolbox.sparse_autoencoder.autoencoder import  Autoencoder
# from NeuralNetworksToolbox.deep_network.dbn import DBNClassifier
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import train_test_split
from nolearn.dbn import DBN
# import mSDA
# from mSDA import linear_msda
# from mSDA.linear_msda import mSDA
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from breze.learn.autoencoder import DenoisingAutoEncoder, ContractiveAutoEncoder
from breze.learn.sparsefiltering import SparseFiltering as br_SpFilt
from breze.learn.lde import LinearDenoiser

import sparse_filtering #.SparseFiltering

# import sparseFiltering

# from dbnlib.dbn import DBN as dbnlibDBN

def getCVScore(predictor,data,labels,cv=5):
    scores = cross_validation.cross_val_score(predictor,X=data, y=labels,cv=cv,n_jobs=-1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


if __name__ == "__main__":
    # loc_train = "kaggle_forest\\train.csv"
    # loc_test = "kaggle_forest\\test.csv"
    # loc_submission = "kaggle_forest\\kaggle.forest.submission.csv"
    loc_submission = "./kaggle_forest/kaggle.forest.submission.csv"
    loc_train = r"./data/train.csv"

    loc_pred = r"./data/test.csv"

    loc_unlabel=r"./data/miniTest.csv"
    df_unlabeled=pd.read_csv(loc_unlabel)


    df_train = pd.read_csv(loc_train)
    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

    #Orig# X_train = df_train[feature_cols]
    X = df_train[feature_cols].values #.astype(float)
    X_unlabeled = df_unlabeled[feature_cols].values
    # X= np.asarray(X)

    # X_train = df_train[feature_cols].values.astype(float)
    # X_train = np.asarray( X_train )
    # X=X_train

    y = df_train['Cover_Type'].values #.astype(int)
    y = np.asarray( y )

    # print(type(y[0]), type(y))
    # print("y",set(y))

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)

    'Feature dimensionality ? :'
    dim = len(feature_cols)
    # featCount = len(feature_cols)
    # print("featCount",featCount)
    # print(feature_cols)

    # sda = SdA(
    #         numpy_rng=numpy_rng,
    #         n_ins=featCount,
    #         hidden_layers_sizes=[200, 100, 30],
    #         n_outs=7    )


    '''
    clf = ensemble.RandomForestClassifier(n_estimators = 330 , n_jobs = -1)
    print("Full Training+test CV:")
    print(cross_val_score(clf,X, y).mean())
    # cross_validation.cross_val_score(clf,X=X, y=y,cv=3)

    # print("Full Training+test data CV:")
    # getCVScore(clf,X,y)

    print("subset Training data CV:")
    print(getCVScore(clf,X_train, y_train).mean())
    '''
    clf = RandomForestClassifier(n_estimators = 250, n_jobs = -1,max_features=23)

    # print("Full Training+test data CV:")
    # getCVScore(clf,X,y)

    # print("subset Training data CV:")
    # getCVScore(clf,X_train, y_train)


    sp1 = sparse_filtering.SparseFiltering( n_features=160,maxfun=855)
    sp2 = sparse_filtering.SparseFiltering( n_features=45,maxfun=255)
    # sp3 = sparse_filtering.SparseFiltering( n_features=18,maxfun=465)


    br_sp1 = br_SpFilt(n_inpt=dim, n_feature=160)

    sp1.fit(X_unlabeled)
    X_sp1=sp1.transform(X)

    brsp1.fit(X_unlabeled)
    X_brsp1=brsp1.transform(X)

    # sp2.fit(X_sp1)
    # sp2.fit(sp1.transform(X_unlabeled))
    # X_sp2=sp2.transform(X_sp1)

    # sp3.fit(X_sp2)
    # X_sp3=sp3.transform(X_sp2)

    print("Sparse filtering 1  score:")
    getCVScore(clf,X_sp1, y)

    # print("Sparse filtering 2  score:")
    # getCVScore(clf,X_sp2, y)
    # print("Sparse filtering 3  score:")
    # getCVScore(clf,X_sp3, y)
    print("Breze Sparse filtering 1 score:")
    getCVScore(clf,X_brsp1, y)


    ld2 = LinearDenoiser(0.2)
    ld2.fit(X_unlabeled)
    X_ld1=brsp1.transform(X)
    print("linear denoised CV:")
    getCVScore(clf,X_ld1, y)

    dA1 = DenoisingAutoEncoder(n_inpt=dim, n_hidden=160)
    dA1.fit(brsp1.transform(X_unlabeled))
    X_unlabeled_AE = dA1.transform(brsp1.transform(X_unlabeled))

    print("linear +DenoisingAutoEncoder1 denoised CV:")
    getCVScore(clf,dA1.transform(X_ld1), y)



    print("Layer wise sparse filtering concat")
    # from sklearn.pipeline import FeatureUnion
    # combined_features = FeatureUnion([("s1",sp1),("s2",sp2),("s3",sp3)],n_jobs=-1)
    # X_combFeatures= combined_features.fit(X, y).transform(X)

    # X_combFeatures = np.hstack(X,X_sp3)
    X_combFeatures =np.hstack((X,X_sp1))
    # X_combFeatures =np.hstack((X_combFeatures,X_sp2))
    getCVScore(clf,X_combFeatures,y)


    #dbn = DBN(input=X_train, label=y, n_ins=featCount, hidden_layer_sizes=[64,50,30],n_outs=len(X_train))


   ## #algorithm='l-bfgs'
    # ae = Autoencoder(algorithm='l-bfgs',verbose=False,
    #     max_iter=1000,n_hidden=250)

    # ae_features = ae.fit_transform(X)
    # print()
    # print(('Training Classification on raw score: ', clf.score(X, y)))
    # print()
    # clf.fit(ae_features, y)
    # print()
    # print(('Training Classification on Sparse Autoencoder features score: ', clf.score(ae_features, y)))
    # print()
    # clf = DBNClassifier(n_hidden=[100,50, 30], algorithm='sgd',
    #                     activation='logistic').fit(X, y)
    # print(('Deep Belief Network score: ', clf.score(X, y)))


#NoLearn DBN"
    # dbn = DBN(
    #     [X.shape[1], 220, 7],
    #     learn_rates = 0.3,
    #     learn_rate_decays = 0.9,
    #     epochs = 50,
    #     verbose = 1)
    dbn = DBN(
        [-1, 250,40,90, -1],
        learn_rates = 0.3,
        learn_rate_decays = 0.95,
        epochs = 325,
        verbose = 0,epochs_pretrain=150)


    ld1 = LinearDenoiser(0.1)
    ld3 = LinearDenoiser(0.3)

    # 'dbnlib DBN - https://github.com/RBMLibrary/DBN-Library/blob/master/dbnlib/dbn.py#L32'

    # dbn.fit(X_train, y_train)
    # print("Fitting nolearn DBN - ")
    # dbn.fit(X_combFeatures, y)
    # dbn.fit(X, y)

    # getCVScore(X_sp1, y)
    print("nolearn DBN predict score:")
    cross_val_score(dbn,X, y)

    print("denoised + nolearn DBN predict score:")
    getCVScore(dbn,dA1.transform(X_ld1), y)
    '''
    y_pred = dbn.predict(X_test)
    print ("Accuracy:", zero_one_loss(y_test, y_pred))
    print ("Classification report:")
    print (classification_report(y_test, y_pred))

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier()
    # clf.fit(X, y)
    '''
    '''
    # clf.fit(X_train, y)
    print("Raw RF CV:")
    cross_val_score(estimator=clf,X=X_train,y=y)
    '''


    df_test = pd.read_csv(loc_pred)
    X_test = df_test[feature_cols]
    test_ids = df_test['Id']


    with open(loc_submission, "wt") as outfile:
        outfile.write("Id,Cover_Type\n")
            for e, val in enumerate(list(clf.predict(X_test))):
                outfile.write("%s,%s\n"%(test_ids[e],val))
                # outfile.write(test_ids[e]+","+val+" \n")
