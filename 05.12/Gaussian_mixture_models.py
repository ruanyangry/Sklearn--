# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

print("#---------------------------------------#")
print("          Gaussian mixture models        ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets.samples_generator import make_blobs
X,y_true=make_blobs(n_samples=400,centers=4,cluster_std=0.60,random_state=0)
X=X[:,::-1]

print(" Plot the data with K Means Labels")

from sklearn.cluster import KMeans
kmeans=KMeans(4,random_state=0)
labels=kmeans.fit(X).predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap='viridis')
plt.show()

from scipy.spatial.distance import cdist

def plot_kmeans(kmeans,X,n_clusters=4,rseed=0,ax=None):
	labels=kmeans.fit_predict(X)
	# Plot the input data
	ax=ax or plt.gca()
	ax.axis('equal')
	ax.scatter(X[:,0],X[:,1],c=labels,s=40,cmap="viridis",zorder=2)
	# Plot the representation of the Kmeans model
	centers=kmeans.cluster_centers_
	radii=[cdist(X[labels==i],[center]).max() for i,center in enumerate(centers)]
	for c,r in zip(centers,radii):
		ax.add_patch(plt.Circle(c,r,fc='#CCCCCC',lw=3,alpha=0.5,zorder=1))
	
kmeans=KMeans(n_clusters=4,random_state=0)
plot_kmeans(kmeans,X)
plt.show()

rng=np.random.RandomState(13)
X_stretched=np.dot(X,rng.randn(2,2))

kmeans=KMeans(n_clusters=4,random_state=0)
plot_kmeans(kmeans,X_stretched)
plt.show()

print("#---------------------------------------#")
print(" Generalizing E-M: Gaussian mixture models")
print("    expectation-maximization approach    ")
print("#---------------------------------------#")
print("\n")

from sklearn.mixture import GMM
gmm=GMM(n_components=4).fit(X)
labels=gmm.predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap="viridis")
plt.show()

probs=gmm.predict_proba(X)
print(probs[:5].round(3))

from matplotlib.patches import Ellipse

def draw_ellipse(position,covariance,ax=None,**kwargs):
	'''
	Draw a ellipse with a give position and covariance
	'''
	ax=ax or plt.gca()
	
	# Convert covariance to principal axes
	if covariance.shape == (2,2):
		U,s,Vt=np.linalg.svd(covariance)
		angle=np.degrees(np.arctan2(U[1,0],U[0,0]))
		width,height=2*np.sqrt(s)
	else:
		angle=0
		width,height=2*np.sqrt(covariance)
		
	# Draw the Ellipse
	for nsig in range(1,4):
		ax.add_patch(Ellipse(position,nsig*width,nsig*height,angle,**kwargs))
		
def plot_gmm(gmm,X,label=True,ax=None):
	ax=ax or plt.gca()
	labels=gmm.fit(X).predict(X)
	if label:
		ax.scatter(X[:,0],X[:,1],c=labels,s=40,cmap="viridis",zorder=2)
	else:
		ax.scatter(X[:,0],X[:,1],s=40,zorder=2)
	ax.axis('equal')
	
	w_factor=0.2/gmm.weights_.max()
	for pos,covar,w in zip(gmm.means_,gmm.covars_,gmm.weights_):
		draw_ellipse(pos,covar,alpha=w*w_factor)
		
gmm=GMM(n_components=4,random_state=42)
plot_gmm(gmm,X)
plt.show()

gmm=GMM(n_components=4,covariance_type='full',random_state=42)
plot_gmm(gmm,X_stretched)
plt.show()

from sklearn.datasets import make_moons
Xmoon,ymoon=make_moons(200,noise=.05,random_state=0)
plt.scatter(Xmoon[:,0],Xmoon[:,1])
plt.show()

gmm2=GMM(n_components=2,covariance_type="full",random_state=0)
plot_gmm(gmm2,Xmoon)
plt.show()

gmm16=GMM(n_components=16,covariance_type="full",random_state=0)
plot_gmm(gmm16,Xmoon,label=False)
plt.show()

Xnew=gmm16.sample(400,random_state=42)
plt.scatter(Xnew[:,0],Xnew[:,1])
plt.show()

n_components=np.arange(1,21)
models=[GMM(n,covariance_type='full',random_state=0).fit(Xmoon) for n in n_components]

plt.plot(n_components,[m.bic(Xmoon) for m in models],label="BIC")
plt.plot(n_components,[m.aic(Xmoon) for m in models],label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")
plt.show()

print("#---------------------------------------#")
print("    Test GMM model to hand write number  ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_digits
digits=load_digits()
print(digits.data.shape)

def plot_digits(data):
	fig,ax=plt.subplots(10,10,figsize=(8,8),subplot_kw=dict(xticks=[],yticks=[]))
	fig.subplots_adjust(hspace=0.05,wspace=0.05)
	for i,axi in enumerate(ax.flat):
		im=axi.imshow(data[i].reshape(8,8),cmap="binary")
		im.set_clim(0,16)
		
plot_digits(digits.data)
plt.show()

# PCA
from sklearn.decomposition import PCA
pca=PCA(0.99,whiten=True)
data=pca.fit_transform(digits.data)
print(data.shape)

print("#---------------------------------------#")
print("    Decided how many components for GMM  ")
print("#---------------------------------------#")
print("\n")

n_components=np.arange(50,210,10)
models=[GMM(n,covariance_type="full",random_state=0) for n in n_components]
aics=[model.fit(data).aic(data) for model in models]
plt.plot(n_components,aics)
plt.show()

gmm=GMM(110,covariance_type='full',random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new=gmm.sample(100,random_state=0)
print(data_new.shape)

digits_new=pca.inverse_transform(data_new)
plot_digits(digits_new)
plt.show()
