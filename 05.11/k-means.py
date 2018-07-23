# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

print("#---------------------------------------#")
print("            k-means classify             ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets.samples_generator import make_blobs
X,y_true=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
plt.scatter(X[:,0],X[:,1],s=50)
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50,cmap='viridis')
centers=kmeans.cluster_centers_
print("centers position")
print(centers)
print("\n")
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5)
plt.show()

print("#---------------------------------------#")
print("           E-step and M-step             ")
print("#---------------------------------------#")
print("\n")

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X,n_clusters,rseed=2):
	# 1 Randomly select clusters
	rng=np.random.RandomState(rseed)
	i=rng.permutation(X.shape[0])[:n_clusters]
	centers=X[i]
	
	while True:
		# 2a Assign labels based on closees center
		labels=pairwise_distances_argmin(X,centers)
		# 2b Find new centers from means of poins
		new_centers=np.array([X[labels==i].mean(0) for i in range(n_clusters)])
		# 2c check for convergence
		if np.all(centers==new_centers):
			break
		centers=new_centers
	return centers,labels
	
centers,labels=find_clusters(X,4)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.show()

print("change the seed number")
print("\n")

centers,labels=find_clusters(X,4,rseed=0)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.show()

print("test the clusters number")
print("\n")

labels=KMeans(6,random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.show()

print("#---------------------------------------#")
print("         Non-linear boundaried           ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import make_moons
X,y=make_moons(200,noise=.05,random_state=0)

labels=KMeans(2,random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap="viridis")
plt.show()

print("#---------------------------------------#")
print("         kernelized k-means           ")
print("#---------------------------------------#")
print("\n")

from sklearn.cluster import SpectralClustering

model=SpectralClustering(n_clusters=2,affinity="nearest_neighbors",assign_labels="kmeans")
labels=model.fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap="viridis")
plt.show()
