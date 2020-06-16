# Title



```python
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

```

```python
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# #############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')
```

    Downloading 20news dataset. This may take a few minutes.
    2018-12-15 19:53:57,797 INFO Downloading 20news dataset. This may take a few minutes.
    Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)
    2018-12-15 19:53:57,800 INFO Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)
    

    Automatically created module for IPython interactive environment
    Usage: ipykernel_launcher.py [options]
    
    Options:
      -h, --help            show this help message and exit
      --report              Print a detailed classification report.
      --chi2_select=SELECT_CHI2
                            Select some number of features using a chi-squared
                            test
      --confusion_matrix    Print the confusion matrix.
      --top10               Print ten most discriminative terms per class for
                            every classifier.
      --all_categories      Whether to use all categories or not.
      --use_hashing         Use a hashing vectorizer.
      --n_features=N_FEATURES
                            n_features when using the hashing vectorizer.
      --filtered            Remove newsgroup information that is easily overfit:
                            headers, signatures, and quoting.
    
    Loading 20 newsgroups dataset for categories:
    ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data loaded
    

```python
# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()
```

    2034 documents - 3.980MB (training set)
    1353 documents - 2.867MB (test set)
    4 categories
    
    

```python
import re 
```

```python
s = "From ABD to XYZ hello From LMN to CDF"
pattern = "From(.*?)to"

substring = re.findall(pattern, s)
print(substring)
```

    [' ABD ', ' LMN ']
    

```python
s = "From ABD to XYZ hello From LMN to CDF"
pattern = "From(.*?)to\s([A-Z]*)"

substring = re.findall(pattern, s)
print(substring)
```

    [(' ABD ', 'XYZ'), (' LMN ', 'CDF')]
    

```python
!pip install nbdev

```

    Collecting nbdev
      Downloading nbdev-0.2.18-py3-none-any.whl (45 kB)
    Collecting fastscript
      Downloading fastscript-0.1.4-py3-none-any.whl (11 kB)
    Requirement already satisfied: packaging in c:\users\akshara\anaconda3\lib\site-packages (from nbdev) (17.1)
    Requirement already satisfied: pyyaml in c:\users\akshara\anaconda3\lib\site-packages (from nbdev) (3.13)
    Requirement already satisfied: nbformat>=4.4.0 in c:\users\akshara\anaconda3\lib\site-packages (from nbdev) (4.4.0)
    Collecting nbconvert>=5.6.1
      Downloading nbconvert-5.6.1-py2.py3-none-any.whl (455 kB)
    Requirement already satisfied: six in c:\users\akshara\appdata\roaming\python\python37\site-packages (from packaging->nbdev) (1.14.0)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\users\akshara\anaconda3\lib\site-packages (from packaging->nbdev) (2.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\users\akshara\anaconda3\lib\site-packages (from nbformat>=4.4.0->nbdev) (2.6.0)
    Requirement already satisfied: ipython-genutils in c:\users\akshara\anaconda3\lib\site-packages (from nbformat>=4.4.0->nbdev) (0.2.0)
    Requirement already satisfied: traitlets>=4.1 in c:\users\akshara\anaconda3\lib\site-packages (from nbformat>=4.4.0->nbdev) (4.3.2)
    Requirement already satisfied: jupyter-core in c:\users\akshara\anaconda3\lib\site-packages (from nbformat>=4.4.0->nbdev) (4.4.0)
    Requirement already satisfied: testpath in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (0.3.1)
    Requirement already satisfied: jinja2>=2.4 in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (2.11.2)
    Requirement already satisfied: bleach in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (2.1.4)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (0.2.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (0.8.3)
    Requirement already satisfied: defusedxml in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (0.5.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (1.4.2)
    Requirement already satisfied: pygments in c:\users\akshara\anaconda3\lib\site-packages (from nbconvert>=5.6.1->nbdev) (2.2.0)
    Requirement already satisfied: decorator in c:\users\akshara\anaconda3\lib\site-packages (from traitlets>=4.1->nbformat>=4.4.0->nbdev) (4.3.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\akshara\anaconda3\lib\site-packages (from jinja2>=2.4->nbconvert>=5.6.1->nbdev) (1.0)
    Requirement already satisfied: html5lib!=1.0b1,!=1.0b2,!=1.0b3,!=1.0b4,!=1.0b5,!=1.0b6,!=1.0b7,!=1.0b8,>=0.99999999pre in c:\users\akshara\anaconda3\lib\site-packages (from bleach->nbconvert>=5.6.1->nbdev) (1.0.1)
    Requirement already satisfied: webencodings in c:\users\akshara\anaconda3\lib\site-packages (from html5lib!=1.0b1,!=1.0b2,!=1.0b3,!=1.0b4,!=1.0b5,!=1.0b6,!=1.0b7,!=1.0b8,>=0.99999999pre->bleach->nbconvert>=5.6.1->nbdev) (0.5.1)
    Installing collected packages: fastscript, nbconvert, nbdev
      Attempting uninstall: nbconvert
        Found existing installation: nbconvert 5.4.0
        Uninstalling nbconvert-5.4.0:
          Successfully uninstalled nbconvert-5.4.0
    Successfully installed fastscript-0.1.4 nbconvert-5.6.1 nbdev-0.2.18
    

    WARNING: You are using pip version 20.0.2; however, version 20.1.1 is available.
    You should consider upgrading via the 'c:\users\akshara\anaconda3\python.exe -m pip install --upgrade pip' command.
    
