~~~~~~~~~~~~~~~~~~~~~~~~~~~
C S 519/487: Semester Project
Stage 5
    William Baker
    Aaron Hudson
    Troy McMillan
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a result of the tremendous load on shelters, and the animals that reside
there, this ML problem was meant to try to gain insight into or even understand
trends in the data representing what happens to these animals. Millions of
animals are euthanized each year and analyzing data the way that we have here
can go towards improving or even saving the lives of a large number of these
animals. At this stage we have implemented a multilayer neural network for the
porpose of trying to process some of the more feature rich datasets provided by
the Austin Animal Center, the same organization that has provided all of our
data.


Sample commands
```
    python main.py perceptron data\train_conversion_lowfeat_sansheader.csv
    python main.py perceptron data\train_conversion_extfeat_sansheader.csv

    python nn.py -f data\train_conversion_extfeat_withheader.csv
    python nn.py -f data\train_conversion_extfeat_withheader.csv -e 20
```

%sh> python main.py -h
usage: main.py [-h] [--builtin] [--fetch] [--nosplit] [--maxrows MAXROWS] [--splitseed SPLITSEED] [--splitsize SPLITSIZE] [--nostandard]
               [--reduce {none,pca,lda,kpca}] [--redcomps REDCOMPS] [--kpca_kernel {linear,poly,rbf,sigmoid,cosine}] [--randseed RANDSEED]
               [--eta ETA] [--iter ITER] [--C C] [--kernel {linear,rbf,poly,sigmoid}] [--criterion {gini,entropy}] [--maxdepth MAXDEPTH]
               [--neighbors NEIGHBORS] [--knnmetric {minkowski}] [--knnmetricp {1,2}] [--n_est N_EST] [--n_jobs N_JOBS]
               [--min_samples_split MIN_SAMPLES_SPLIT] [--min_samples_leaf MIN_SAMPLES_LEAF] [--bootstrap] [--max_features MAX_FEATURES]
               [--max_samples MAX_SAMPLES]
               {perceptron,svm,dtree,knn,randforest,adaboost,bagging} datafile

positional arguments:
  {perceptron,svm,dtree,knn,randforest,adaboost,bagging}
  datafile              file containing the data, in CSV format (last column is class)

optional arguments:
  -h, --help            show this help message and exit
[...omitted for brevity...]




%sh> python nn.py -h
usage: nn.py [-h] -f CSV [-b BATCH] [-s SIZE] [-e EPOCH]

optional arguments:
  -h, --help  show this help message and exit
  -f CSV      specify *relative* location of input data. (Expects CSV)
  -b BATCH    specify batch size, default is 100
  -s SIZE     specify buffer size, default is 10,000
  -e EPOCH    specify the epoch, default is 10

