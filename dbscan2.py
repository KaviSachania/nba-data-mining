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
eps_ = 0.45
min_samples_ = 5

# #############################################################################
with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
X = genfromtxt( filename, delimiter=',') 
Y = X[1:,7:-1]

# #############################################################################
# Compute DBSCAN

db = DBSCAN(eps=eps_, min_samples=min_samples_).fit(Y)
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

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"  % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels)) 
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

for i in range(len(labels)):
  labels[i]+=1

#############################
####RADAR CHAT###############

feature_from = 0
feature_to = 18
categories = [data[0][feature_from+7],
              data[0][feature_from+7+1],
              data[0][feature_from+7+2],
              data[0][feature_from+7+3],
              data[0][feature_from+7+4],
              data[0][feature_from+7+5],
              data[0][feature_from+7+6],
              data[0][feature_from+7+7],
              data[0][feature_from+7+8],
              data[0][feature_from+7+9],
              data[0][feature_from+7+10],
              data[0][feature_from+7+11],
              data[0][feature_from+7+12],
              data[0][feature_from+7+13],
              data[0][feature_from+7+14],
              data[0][feature_from+7+15],
              data[0][feature_from+7+16],
              data[0][feature_from+7+17],
              data[0][feature_from+7+18],
              data[0][feature_from+7+19],
              data[0][feature_from+7+20],
              data[0][feature_from+7+21],
              data[0][feature_from+7+22],
              data[0][feature_from+7+23]
              ]
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

colors = ['black', 'red', 'blue', 'green','orange','yellow', 'pink', 'brown']

fig = go.Figure()


    
for i in range(len(Y)):
  fig.add_trace(go.Scatterpolar(
        r= Y[i][feature_from:feature_to],
        theta=categories,
        line=dict(color=colors[labels[i]], width=4),
   
        #name= data[i+1][2]
        name = data[i+1][2] +" (" +  data[i+1][-1] + ")",
       
  ))
  



max_val = np.amax(Y[0][feature_from:feature_to])
for i in range(len(Y)):
  if (max_val < np.amax(Y[i][feature_from:feature_to])):
    max_val = np.amax(Y[i][feature_from:feature_to])


# Set title
fig.update_layout(
    title_text="DBSCAN: Guards & Centers & Forwards"
)

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, max_val]
      #range=[0, np.amax(Y[i][5:9])]
    )),
  showlegend=True
)

plot(fig)


######################
####2D PCA

fig = plt.figure(1, figsize=(4, 3))

plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(Y)
Y = pca.transform(Y)

colors = ['black', 'red', 'blue', 'green','orange','yellow', 'pink', 'brown']

for i in range(len(Y)):
    plt.scatter(Y[i, 0], Y[i, 1], c=(colors[labels[i]]), s=50)
    
plt.title('PCA Graph: DBSCAN: Guards & Centers & Forwards' )
plt.xlabel('pc1')
plt.ylabel('pc2')
