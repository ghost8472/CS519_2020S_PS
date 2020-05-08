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
    python main.py -f test.csv
    python main.py -f ../../test.csv -e 20
    python main.py -f ./test.csv -b 50 -s 1000
```

%sh> python main.py -h
usage: main.py [-h] -f CSV [-b BATCH] [-s SIZE] [-e EPOCH]

optional arguments:
  -h, --help  show this help message and exit
  -f CSV      specify *relative* location of input data. (Expects CSV)
  -b BATCH    specify batch size, default is 100
  -s SIZE     specify buffer size, default is 10,000
  -e EPOCH    specify the epoch, default is 10

