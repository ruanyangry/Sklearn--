# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

print("#---------------------------------------#")
print("           k-means on digits             ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_digits

digits=load_digits()
print(digits.data.shape)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=10,random_state=0)
clusters=kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)

fig,ax=plt.subplots(2,5,figsize=(8,3))
centers=kmeans.cluster_centers_.reshape(10,8,8)
for axi,center in zip(ax.flat,centers):
	axi.set(xticks=[],yticks=[])
	axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
	
plt.show()

from scipy.stats import mode

labels=np.zeros_like(clusters)
for i in range(10):
	mask=(clusters==i)
	labels[mask]=mode(digits.target[mask])[0]
	
from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target,labels))

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(digits.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,\
xticklabels=digits.target_names,yticklabels=digits.target_names)
plt.xlabel("True label")
plt.ylabel("predicted label")
plt.show()

from sklearn.manifold import TSNE

# Project the data: this step will take sever seconds

tsne=TSNE(n_components=2,init='random',random_state=0)
digits_proj=tsne.fit_transform(digits.data)

# compute the clusters
kmeans=KMeans(n_clusters=10,random_state=0)
clusters=kmeans.fit_predict(digits_proj)

# Permute the labels

labels=np.zeros_like(clusters)

for i in range(10):
	mask=(clusters == i)
	labels[mask]=mode(digits.target[mask])[0]
	
# compute the accuracy

accuracy_score(digits.target,labels)

print("#---------------------------------------#")
print("   k-means for color compression         ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_sample_image
china=load_sample_image("china.jpg")
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)

print(china.shape)

data=china/255.0
data=data.reshape(427*640,3)
print(data.shape)


def plot_pixels(data,title,colors=None,N=10000):
	if colors is None:
		colors=data
		
	# choose a random subset
	rng=np.random.RandomState(0)
	i=rng.permutation(data.shape[0])[:N]
	colors=colors[i]
	R,G,B=data[i].T
	
	fig,ax=plt.subplots(1,2,figsize=(16,6))
	ax[0].scatter(R,G,color=colors,marker='.')
	ax[0].set(xlabel='Red',ylabel="Green",xlim=(0,1),ylim=(0,1))
	
	ax[1].scatter(R,B,color=colors,marker='.')
	ax[1].set(xlabel='Red',ylabel='Blue',xlim=(0,1),ylim=(0,1))
	
	fig.suptitle(title,size=20)
	
plot_pixels(data,title='Input color space: 16 million possible colors')
plt.show()

import warnings
warnings.simplefilter('ignore')

from sklearn.cluster import MiniBatchKMeans

kmeans=MiniBatchKMeans(16)
kmeans.fit(data)

new_colors=kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data,colors=new_colors,title='Reduced color space: 16 colors')
plt.show()

china_recolored = new_colors.reshape(china.shape)

fig,ax=plt.subplots(1,2,figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title("Original Image",size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image',size=16)
plt.show()
