import scipy.io as sio
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
from scipy import sparse
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


u1=open("C://Users//xp//Desktop//Project @ BD//social_honeypot_icwsm_2011//content_polluters.txt")
u2=open("C://Users//xp//Desktop//Project @ BD//social_honeypot_icwsm_2011//legitimate_users.txt")

uid1=[]
p1=[]
nfing1=[]
nfer1=[]
nt1=[]
ln1=[]
ld1=[]
uid2=[]
p2=[]
nfing2=[]
nfer2=[]
label=[]
nt2=[]
ln2=[]
ld2=[]

for line in u1:
    items=line.strip().split('\t')
    uid1.append(items[0])
    p1.append(str((items[1])))
    nfing1.append(int(items[3]))
    nfer1.append(int(items[4]))
    nt1.append(int(items[5]))
    ln1.append(int(items[6]))
    ld1.append(int(items[7]))


for line in u2:
    items=line.strip().split('\t')
    uid2.append(items[0])
    p2.append(str(items[1]))
    nfing2.append(int(items[3]))
    nfer2.append(int(items[4]))
    nt2.append(int(items[5]))
    ln2.append(int(items[6]))
    ld2.append(int(items[7]))

nfing1=np.array(nfing1)
n1=nfing1.shape
nfing2=np.array(nfing2)
n2=nfing2.shape
nfer1=np.array(nfer1)
nfer2=np.array(nfer2)
nt1=np.array(nt1)
nt2=np.array(nt2)
ln1=np.array(ln1)
ln2=np.array(ln2)
ld1=np.array(ld1)
ld2=np.array(ld2)
nfing=numpy.hstack([nfing1,nfing2])

import datetime
xdates1 = [datetime.datetime.strptime(str(date),'%Y-%b-%d %H:%M:%S') for date in p1]

import matplotlib.pyplot as plt


plt.hist(nfing1, color='b', label='content polluter')
plt.hist(nfing2, color='r', alpha=0.5, label='legitimate user')
plt.title("number of followings")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.legend()
plt.show()



