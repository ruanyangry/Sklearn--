# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

print('#---------------------------------#')
print('   Test manifold learning methods  ')
print('     Isometric mapping (IsoMap)    ')
print(' t-distributed stochastic neighbor embedding (t-SNE)')
print('#---------------------------------#')
print("\n")

from sklearn.datasets import fetch_lfw_people
faces=fetch_lfw_people(min_faces_per_person=30)
print(faces.data.shape)

fig,ax=plt.subplots(4,8,subplot_kw=dict(xticks=[],yticks=[]))
for i,axi in enumerate(ax.flat):
	axi.imshow(faces.images[i],cmap='gray')
	
plt.show()
	
print('#---------------------------------#')
print('  PCA:based explained variance ratio\
decided the optimal dimensions')
print('#---------------------------------#')
print("\n")

from sklearn.decomposition import RandomizedPCA
model=RandomizedPCA(100).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance')
plt.show()

print('#---------------------------------#')
print("          Isomap methods           ")
print('#---------------------------------#')
print("\n")

from sklearn.manifold import Isomap
model=Isomap(n_components=2)
proj=model.fit_transform(faces.data)
print(proj.shape)

# Draw image thumbnails

from matplotlib import offsetbox

def plot_components(data,model,images=None,ax=None,\
thumb_frac=0.05,cmap='gray'):
	ax=ax or plt.gca()
	proj=model.fit_transform(data)
	ax.plot(proj[:,0],proj[:,1],'.k')
	
	if images is not None:
		min_dist_2=(thumb_frac*max(proj.max(0)-proj.min(0)))**2
		shown_images=np.array([2*proj.max(0)])
		for i in range(data.shape[0]):
			dist=np.sum((proj[i]-shown_images)**2,1)
			if np.min(dist) < min_dist_2:
				# don't show points that are too close
				continue
			shown_images=np.vstack([shown_images,proj[i]])
			imagebox=offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i],\
			cmap=cmap),proj[i])
			ax.add_artist(imagebox)
			
fig,ax=plt.subplots(figsize=(10,10))
plot_components(faces.data,model=Isomap(n_components=2),images=faces.images[:,::2,::2])
plt.show()

print('#---------------------------------#')
print(" Visualizing Structure in Digits   ")
print('#---------------------------------#')
print("\n")

from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
print(mnist.data.shape)

fig,ax = plt.subplots(6,8,subplot_kw=dict(xticks=[],yticks=[]))
for i,axi in enumerate(ax.flat):
	axi.imshow(mnist.data[1250*i].reshape(28,28),cmap='gray_r')
	
data=mnist.data[::30]
target=mnist.target[::30]

model=Isomap(n_components=2)
proj=model.fit_transform(data)
plt.scatter(proj[:,0],proj[:,1],c=target,cmap=plt.cm.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

from sklearn.manifold import Isomap

data=mnist.data[mnist.target==1][::4]

fig,ax=plt.subplots(figsize=(10,10))
model=Isomap(n_neighbors=5,n_components=2,eigen_solver='dense')
plot_components(data,model,images=data.reshape((-1,28,28)),ax=ax,thumb_frac=0.05,\
cmap='gray_r')
plt.show()
