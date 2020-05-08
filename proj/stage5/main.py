#!/bin/python3
#written using Python 3.8.1, by William Baker, February-March 2020, for NMSU's  CS-519 course

import argparse
import pandas as pd
import numpy as np
import time
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

aparser = argparse.ArgumentParser()
aparser.add_argument('neurontype', type=str, choices=['perceptron','svm','dtree','knn','randforest','adaboost','bagging'])
aparser.add_argument('datafile', type=str, help="file containing the data, in CSV format (last column is class)")
aparser.add_argument('--builtin', action='store_true', help="use flag to treat the 'datafile' as a scikit-learn data source instead of a file")
aparser.add_argument('--fetch', action='store_true', help="use flag to treat the 'datafile' as a scikit-learn data fetch from OpenML instead of a file")
aparser.add_argument('--nosplit', action='store_true', help="do not split the data between train and test")
aparser.add_argument('--maxrows', default=20000, help='if rows exceed this max, it will crop them')
aparser.add_argument('--splitseed', type=int, default=1, help="the random seed used to shuffle/split the data into train/test")
aparser.add_argument('--splitsize', type=float, default=0.3, help="the amount of data cases used for test cases")
aparser.add_argument('--nostandard', action='store_true', help="do not standardize the data")
aparser.add_argument('--reduce', type=str, default="none", choices=["none","pca","lda","kpca"], help="feature/dimension reduction algorithm to use")
aparser.add_argument('--redcomps', type=int, default=2, help="Feature reduction - number of components/eigenvectors/features to keep")
aparser.add_argument('--kpca_kernel', type=str, default="rbf", choices=["linear", "poly", "rbf", "sigmoid", "cosine"], help="Feature reduction - KPCA - kernel to use")
aparser.add_argument('--randseed', type=int, default=1, help="All neurons - set the random seed used for initial state values")
aparser.add_argument('--eta', type=float, default=0.1, help="Perceptron & AdaBoost option - the learning factor to pass to the Perceptron")
aparser.add_argument('--iter', type=int, default=1000, help="Perceptron option - how many epochs of learning")
aparser.add_argument('--C', type=float, default=1.0, help="SVM option - the penalty for miscalculation")
aparser.add_argument('--kernel', type=str, default='rbf', choices=['linear','rbf','poly','sigmoid'], help="SVM option - whether to use linear, or RBF kernel")
aparser.add_argument('--criterion', type=str, default='gini', choices=['gini','entropy'], help="Decision Tree & Random Forest option - what impurity criterion to use")
aparser.add_argument('--maxdepth', type=int, default=4, help="Decision Tree & Random Forest option - maximum depth")
aparser.add_argument('--neighbors', type=int, default=5, help="K Nearest Neighbors option - number of neighbors to find ('k')")
aparser.add_argument('--knnmetric', type=str, default='minkowski', choices=['minkowski'], help="K Nearest Neighbors option - distance calculation metric")
aparser.add_argument('--knnmetricp', type=int, default=2, choices=[1,2], help="K Nearest Neighbors option - when using minkowski, 1=Manhattan distance, 2=Euclidean")
aparser.add_argument('--n_est', type=int, default=25, help="Random Forest, AdaBoost, & Bagging option - number of estimators")
aparser.add_argument('--n_jobs', type=int, default=2, help="Random Forest option - number of jobs to run in parallel")
aparser.add_argument('--min_samples_split', type=int, default=2, help="Random Forest option - minimum samples req. for split")
aparser.add_argument('--min_samples_leaf', type=int, default=2, help="Random Forest option - minimum samples leaf")
aparser.add_argument('--bootstrap', action='store_true', help="Random Forest & Bagging option - whether to bootstrap")
aparser.add_argument('--max_features', type=float, default=1.0, help="Bagging option - maximum features (1.0 means all)")
aparser.add_argument('--max_samples', type=float, default=1.0, help="Bagging option - maximum samples (1.0 means all)")

args = aparser.parse_args()


# ============================ File reading  ===========================================================================
print("Accessing data...")

X = None
y = None

data = pd.read_csv(args.datafile, header=None, encoding='utf-8')

X = data.values[:, 0:-1]
y = data.values[:, -1]

# Note: using "median" so that values are known valid.  consider an int enumerated type, a value of 1.3 makes no sense.
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imr.fit(X)
X = imr.transform(X)

print("Shape of data", X.shape)

# split the data into train and test data sets
if args.nosplit:
    X_train = X
    X_test = X
    y_train = y
    y_test = y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.splitsize, random_state=args.splitseed, stratify=y)

# standardize the inputs
if args.nostandard:
    X_train_std = X_train
    X_test_std = X_test
else:
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


# use the latest X, y values from above transformations/standardizations/reductions
X_train_use = X_train_std
X_test_use = X_test_std
y_train_use = y_train
y_test_use = y_test

# =============================== Neuron initialization ================================================================
print("Initializing neuron...")

base_est = None
if args.neurontype == 'adaboost' or args.neurontype == 'bagging':
    base_est = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.maxdepth, random_state=args.randseed)

neuron = None
if args.neurontype == 'perceptron':
    neuron = Perceptron(eta0=args.eta, tol=None, max_iter=args.iter, random_state=args.randseed)
elif args.neurontype == 'svm':
    neuron = SVC(kernel=args.kernel, C=args.eta, random_state=args.randseed)
elif args.neurontype == 'dtree':
    neuron = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.maxdepth, random_state=args.randseed)
elif args.neurontype == 'knn':
    neuron = KNeighborsClassifier(n_neighbors=args.neighbors, metric=args.knnmetric, p=args.knnmetricp)
elif args.neurontype == 'randforest':
    neuron = RandomForestClassifier(criterion=args.criterion, n_estimators=args.n_est, n_jobs=args.n_jobs,
                                    min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
                                    bootstrap=args.bootstrap, max_depth=args.maxdepth, random_state=args.randseed)
elif args.neurontype == 'adaboost':
    neuron = AdaBoostClassifier(base_estimator=base_est, n_estimators=args.n_est, learning_rate=args.eta, random_state=args.randseed)
elif args.neurontype == 'bagging':
    neuron = BaggingClassifier(base_estimator=base_est, n_estimators=args.n_est, n_jobs=args.n_jobs,
                               max_samples=args.max_samples, max_features=args.max_features
                               , bootstrap=args.bootstrap, random_state=args.randseed)
else:
    print("ERROR: 'neurontype' selected is not supported")
    exit()


# ================================ Learning (time consuming) ===========================================================
print("Starting learning...")
before = time.time()

neuron.fit(X_train_use, y_train_use)

after = time.time()
print("Time to learn: ", (after-before), "seconds")


# ================================ Testing =============================================================================
print("Starting testing...")
before = time.time()

y_train_pred = neuron.predict(X_train_use)
y_test_pred = neuron.predict(X_test_use)
print("Training Accuracy: ", accuracy_score(y_true=y_train_use, y_pred=y_train_pred))
print("Training Precision: ",precision_score(y_true=y_train_use, y_pred=y_train_pred,average='macro'))
print("Training Recall: ",recall_score(y_true=y_train_use, y_pred=y_train_pred,average='macro'))
print("Training F1: ",f1_score(y_true=y_train_use, y_pred=y_train_pred,average='macro'))
print("    --------")
print("Testing Accuracy: ", accuracy_score(y_true=y_test_use, y_pred=y_test_pred))
print("Testing Precision: ", precision_score(y_true=y_test_use, y_pred=y_test_pred,average='macro'))
print("Testing Recall: ", recall_score(y_true=y_test_use, y_pred=y_test_pred,average='macro'))
print("Testing F1: ", f1_score(y_true=y_test_use, y_pred=y_test_pred,average='macro'))

after = time.time()
print("Time to test: ", (after-before), "seconds")






