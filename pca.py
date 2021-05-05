import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import csv
from sklearn import decomposition
from sklearn import datasets

# ##############################################INPUT#########################
filename = 'stats_raw_position_GCF.csv'
n_components = 5

###############################################
with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
X = genfromtxt( filename, delimiter=',') 
Y = X[1:,7:-1]

#########################PCA

fig = plt.figure(1, figsize=(4, 3))

plt.cla()
pca = decomposition.PCA(n_components)
pca.fit(Y)
Y = pca.transform(Y)
###############################################DBSCAN
db = DBSCAN(eps=0.4, min_samples= 5).fit(Y)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

###########################################
########## % CORRECTNESS###################
number_of_G = 0
number_of_C = 0
number_of_F = 0
for i in range(len(data)):
    if data[i][-1] == 'G':
      number_of_G += 1
    if data[i][-1] == 'C':
      number_of_C += 1   
    if data[i][-1] == 'F':
      number_of_F += 1   
      
      
correct_G = 0
correct_C = 0
correct_F = 0
for i in range(len(labels)):
    if data[i+1][-1] == 'G' and labels[i]==0:
      correct_G += 1    
    if data[i+1][-1] == 'C' and labels[i]==1:
      correct_C += 1
    if data[i+1][-1] == 'F' and labels[i]==2:
      correct_F += 1

print('Total number of points: %d' % len(Y))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Guard: Predicted % Correct: " , 100*correct_G/number_of_G)
print("Center: Predicted % Correct: " , 100*correct_C/number_of_C)
print("Forward: Predicted % Correct: " , 100*correct_F/number_of_F)
###########################################




colors = [ 'red', 'blue', 'green','orange','yellow','pink','black']

for i in range(len(Y)):
    plt.scatter(Y[i, 0], Y[i, 1], c=(colors[labels[i]]), s=50)
    
plt.title('PCA Graph: K-Means: Guards & Centers & Forwards' )
plt.xlabel('pc1')
plt.ylabel('pc2')

