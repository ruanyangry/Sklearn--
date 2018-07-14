#_*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print("#---------------------------------------#")
print("         two-dimensional data            ")
print("#---------------------------------------#")
print("\n")

rng=np.random.RandomState(1)
X=np.dot(rng.rand(2,2),rng.randn(2,200)).T
plt.scatter(X[:,0],X[:,1])
plt.axis('equal')
plt.show()

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
print(pca.fit(X))

print(pca.components_)
print(pca.explained_variance_)

def draw_vector(v0,v1,ax=None):
	ax=ax or plt.gca()
	arrowprops=dict(arrowstyle='->',linewidth=2,\
	shrinkA=0,shrinkB=0)
	ax.annotate('',v1,v0,arrowprops=arrowprops)
	
# Plot data
plt.scatter(X[:,0],X[:,1],alpha=0.2)
for length,vector in zip(pca.explained_variance_,pca.components_):
	v=vector*3*np.sqrt(length)
	draw_vector(pca.mean_,pca.mean_+v)
	
plt.axis('equal')
plt.show()

print("#---------------------------------------#")
print("      Dimensionality reduction           ")
print("#---------------------------------------#")
print("\n")

pca=PCA(n_components=1)
pca.fit(X)
X_pca=pca.transform(X)

print("Original shape: ",X.shape)
print("Transformed shape: ",X_pca.shape)

X_new=pca.inverse_transform(X_pca)
plt.scatter(X[:,0],X[:,1],alpha=0.2)
plt.scatter(X_new[:,0],X_new[:,1],alpha=0.8)
plt.axis('equal')
plt.show()