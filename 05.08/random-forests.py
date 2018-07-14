# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

print("#---------------------------------#")
print(' Decision Trees and Random Forests ')
print("#---------------------------------#")
print("\n")

from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=300,centers=4,random_state=0,cluster_std=1.0)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier().fit(X,y)

# Display results

def visualize_classifier(model,X,y,ax=None,cmap='rainbow'):
	ax=ax or plt.gca()
	
	# Plot the training point
	ax.scatter(X[:,0],X[:,1],c=y,s=30,cmap=cmap,clim=(y.min(),y.max()),zorder=3)
	ax.axis('tight')
	ax.axis('off')
	xlim=ax.get_xlim()
	ylim=ax.get_ylim()
	
	# fit the estimator
	model.fit(X,y)
	xx,yy=np.meshgrid(np.linspace(*xlim,num=200),np.linspace(*ylim,num=200))
	Z=model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
	
	# create a color plot with the results
	n_classes=len(np.unique(y))
	contours=ax.contourf(xx,yy,Z,alpha=0.3,levels=np.arange(n_classes+1)-0.5,cmap=cmap,\
	clim=(y.min(),y.max()),zorder=1)
	ax.set(xlim=xlim,ylim=ylim)
	
visualize_classifier(DecisionTreeClassifier(),X,y)
plt.show()

print("#---------------------------------#")
print(' Ensembles of Estimators Random Forests ')
print("       Bagging Classifier          ")
print("#---------------------------------#")
print("\n")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree=DecisionTreeClassifier()
bag=BaggingClassifier(tree,n_estimators=100,max_samples=0.8,random_state=1)
bag.fit(X,y)
visualize_classifier(bag,X,y)
plt.show()

print("#---------------------------------#")
print('      RandomForestsClassifier      ')
print("#---------------------------------#")
print("\n")

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100,random_state=0)
visualize_classifier(model,X,y)
plt.show()

print("#---------------------------------#")
print('    Random Forests Regression      ')
print("#---------------------------------#")
print("\n")

rng=np.random.RandomState(42)
x=10*rng.rand(200)

def model(x,sigma=0.3):
	fast_oscillation=np.sin(5*x)
	slow_oscillation=np.sin(0.5*x)
	noise=sigma*rng.randn(len(x))
	return fast_oscillation+slow_oscillation+noise
	
y=model(x)
plt.errorbar(x,y,0.3,fmt='o')
plt.show()

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(200)
forest.fit(x[:,None],y)

xfit=np.linspace(0,10,1000)
yfit=forest.predict(xfit[:,None])
ytrue=model(xfit,sigma=0)

plt.errorbar(x,y,0.3,fmt='o',alpha=0.5)
plt.plot(xfit,yfit,'-r')
plt.plot(xfit,ytrue,'-k',alpha=0.5)
plt.show()
