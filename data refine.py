__author__ = 'xp'
import sys
from collections import defaultdict
from operator import itemgetter
import nltk
from nltk import word_tokenize
import codecs
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import distance_metrics
import numpy, scipy.io
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist
import itertools
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import distance_metrics
import numpy, scipy.io
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist
import itertools
from scipy.spatial.distance import squareform
import pandas as pd

matfn = 'C:\\Users\\xp\\Desktop\\Project @ BD\\data1.mat'
data = sio.loadmat(matfn)
Xp=data['X']
labelp=data["label"]
uid=data['uid']

[n,f]=Xp.shape

with open ('C:\\Users\\xp\\Desktop\\Project @ BD\\dictionary.txt', "r") as myfile:
    dic=myfile.readlines()


#################Removing features with low variance¶
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
a1=sel.fit_transform(Xp)

##################

#############################################fpr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import VarianceThreshold


from sklearn.feature_selection import SelectFpr

a1=SelectFpr(chi2, 0.01).fit_transform(Xp, labelp)
a1.shape

#######################################mrmr
from mrmr import mixedmrmr


#####################################33correlation/coefficient
from sklearn.svm import LinearSVC
a1 = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(Xp, labelp)
a1.shape
#########################################################################################################################
#####################Cross Validation
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(a1 ,labelp, test_size=0.2, random_state=0)

#########################################################################################################################
################################################SVM
from sklearn import svm
clf = svm.SVC(kernel='linear').fit(X_train, y_train[:,0])
clf_score=clf.score(X_test, y_test)
clf_score
clf1 = svm.SVC().fit(X_train, y_train[:,0])
clf1_score=clf1.score(X_test, y_test)
clf1_score

###############################################33 LogisticRegression
from sklearn.linear_model import LogisticRegression
clf2=LogisticRegression(penalty='l1').fit(X_train, y_train[:,0])
clf2_score=clf2.score(X_test, y_test)
clf2_score
y_new=clf2.predict(X_test)

#########################naive Bayes---
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = cross_validation.train_test_split(a1 ,labelp, test_size=0.2, random_state=0)
clf = MultinomialNB().fit(X_train, y_train[:,0])
clf_score=clf.score(X_test, y_test)


from sklearn.naive_bayes import BernoulliNB
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xp ,labelp, test_size=0.2, random_state=0)
clf = BernoulliNB().fit(X_train, y_train[:,0])
clf_score=clf.score(X_test, y_test)

##############################33NearestCentroid
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid().fit(X_train, y_train[:,0])
clf_score=clf.score(X_test, y_test)
y_new=clf.scorepredict(X_test)



######################################################################embeding  plot


similarities = euclidean_distances(X_train)
scipy.io.savemat('C:\\Users\\xp\\Desktop\\Project @ BD\\similarities(log-coe).mat', mdict={'similarities':similarities})
mds = manifold.MDS(n_components=2, max_iter=500,eps=1e-6, n_jobs=1,dissimilarity=precomputed1)
pos = mds.fit(similarities).embedding_
scipy.io.savemat('C:\\Users\\xp\\Desktop\\Project @ BD\\pos(log-coe).mat', mdict={'pos':pos})

plt.scatter(x=pos[:,0].T,y=pos[:,1].T, c=y_new)

pos2 = mds.fit(similarities,y_test).embedding_
plt.scatter(x=pos2[:,0].T,y=pos2[:,1].T, c=y_new)
