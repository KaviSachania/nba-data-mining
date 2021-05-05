from numpy import genfromtxt
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np


# ##############################################INPUT#########################
filename = 'stats_raw_position_GCF.csv'


# #############################################################################
    
X = genfromtxt( filename, delimiter=',') 
X = X[1:,7:-1]




###############
Sum_of_squared_distances =[] 
k = 10
for i in range(1, k): 
    KM = KMeans(n_clusters = i, n_init = 20, max_iter = 500) 
    KM.fit(X) 
      
    # calculates squared error 
    # for the clustered points 
    Sum_of_squared_distances.append(KM.inertia_)  
    
# plot the cost against K values 
plt.plot(range(1, k), Sum_of_squared_distances, 'go-', color ='g', linewidth ='2') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 
# the point of the elbow is the  
# most optimal value for choosing k 