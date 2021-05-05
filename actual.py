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


# #############################################################################
with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
X = genfromtxt( filename, delimiter=',') 
Y = X[1:,7:-1]

# #############################################################################
labels = np.empty([len(data)-1], dtype=int)

for i in range((len(data)-1)):
  if data[i+1][30] == 'G':
    labels[i] = 1
  if data[i+1][30] == 'C':
    labels[i] = 2
  if data[i+1][30] == 'F':
    labels[i] = 3
  if data[i+1][30] == 'G/F':
    labels[i] = 4
  if data[i+1][30] == 'F/C':
    labels[i] = 5    
# #############################################################################
####RADAR CHAT###############

feature_from = 0
feature_to = 23
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

colors = ['black', 'red', 'blue', 'green','orange','yellow']

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
    title_text="Radar Chart: ACTUAL: All Players and Positions"
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

colors = ['black', 'red', 'blue', 'green','orange','yellow']

for i in range(len(Y)):
    plt.scatter(Y[i, 0], Y[i, 1], c=(colors[labels[i]]), s=50)
    
plt.title('PCA Graph: ACTUAL: All Players and Positions' )
plt.xlabel('pc1')
plt.ylabel('pc2')
