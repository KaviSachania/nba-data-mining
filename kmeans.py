from sklearn.cluster import KMeans
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import csv
from sklearn import decomposition
from sklearn import datasets

# ##############################################INPUT#########################
filename = 'stats_raw_position_GCF.csv'
num_cluster = 3

# #############################################################################
with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
X = genfromtxt( filename, delimiter=',') 
Y = X[1:,7:-1]


####################################################
with open('stats_raw_position_GCF.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
X = genfromtxt('stats_raw_position_GCF.csv', delimiter=',') 
Y = X[1:,7:-1]


####################################################
#####K MEANS########################################
#####K MEANS########################################
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(Y)
kmeans.labels_

labels = kmeans.predict(Y)

kmeans.cluster_centers_
centers = kmeans.cluster_centers_



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
    if data[i+1][-1] == 'G' and labels[i]==1:
      correct_G += 1    
    if data[i+1][-1] == 'C' and labels[i]==0:
      correct_C += 1
    if data[i+1][-1] == 'F' and labels[i]==2:
      correct_F += 1
      
      
print("Guard: Predicted % Correct: " , 100*correct_G/number_of_G)
print("Center: Predicted % Correct: " , 100*correct_C/number_of_C)
print("Forward: Predicted % Correct: " , 100*correct_F/number_of_F)


#############################
####RADAR CHART###############
  
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

colors = ['black', 'blue', 'red', 'green','orange','yellow']

fig = go.Figure()

for i in range(len(labels)):
  labels[i]+=1
    
for i in range(len(Y)):
  fig.add_trace(go.Scatterpolar(
        r= Y[i][feature_from:feature_to],
        theta=categories,
        line=dict(color=colors[labels[i]], width=4),
      
        #name= data[i+1][2]
        name = data[i+1][2] +" (" +  data[i+1][-1] + ")"
  ))


  # fig.add_trace(go.Scatterpolar(
  #       r= Y[i][5:7],
  #       theta=categories,
  #       line=dict(color=colors[y_kmeans[i]], width=4),
      
  #       #name= data[i+1][2]
  #       name = data[i+1][2] +" (" +  data[i+1][-1] + ")"
  # ))


max_val = np.amax(Y[0][feature_from:feature_to])
for i in range(len(Y)):
  if (max_val < np.amax(Y[i][feature_from:feature_to])):
    max_val = np.amax(Y[i][feature_from:feature_to])


# Set title
fig.update_layout(
    title_text="K-Means (k = 4): All Positions"
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

colors = ['black', 'blue', 'red', 'green','orange','yellow']

for i in range(len(Y)):
    plt.scatter(Y[i, 0], Y[i, 1], c=(colors[labels[i]]), s=50)
    
plt.title('PCA Graph: K-Means (k = 4): All Positions' )
plt.xlabel('pc1')
plt.ylabel('pc2')

