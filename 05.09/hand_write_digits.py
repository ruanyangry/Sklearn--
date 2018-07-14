#_*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print("#---------------------------------------#")
print("           Hand write digits             ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
digits=load_digits()
print(digits.data.shape)

# project from 64 to 2

pca=PCA(2)
projected=pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:,0],projected[:,1],c=digits.target,edgecolor='none',\
alpha=0.5,cmap=plt.cm.get_cmap('spectral',10))

plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()

plt.show()

print("#---------------------------------------#")
print("     Choosing the number of components   ")
print("#---------------------------------------#")
print("\n")

pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

print("#---------------------------------------#")
print("          PCA as noise filtering         ")
print("#---------------------------------------#")
print("\n")

def plot_digits(data):
	fig,axes=plt.subplots(4,10,figsize=(10,4),subplot_kw={'xticks':[],'yticks':[]},\
	gridspec_kw=dict(hspace=0.1,wspace=0.1))
	for i,ax in enumerate(axes.flat):
		ax.imshow(data[i].reshape(8,8),cmap='binary',interpolation='nearest',\
		clim=(0,16))
		
plot_digits(digits.data)
plt.show()

print("#---------------------------------------#")
print("               Add data noise            ")
print("#---------------------------------------#")
print("\n")

np.random.seed(42)
noisy=np.random.normal(digits.data,4)
plot_digits(noisy)
plt.show()

print("#---------------------------------------#")
print("            PCA reduce noise             ")
print("#---------------------------------------#")
print("\n")

pca=PCA(0.50).fit(noisy)
print(pca.n_components_)

components=pca.transform(noisy)
filtered=pca.inverse_transform(components)
plot_digits(filtered)
plt.show()
