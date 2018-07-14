# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

fig,ax=plt.subplots(3,5)
for i,axi in enumerate(ax.flat):
	axi.imshow(faces.images[i],cmap='bone')
	axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])
	
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca=RandomizedPCA(n_components=150,whiten=True,random_state=42)
svc=SVC(kernel='rbf',class_weight='balanced')
model=make_pipeline(pca,svc)

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(faces.data,faces.target,random_state=42)

from sklearn.grid_search import GridSearchCV
param_grid={'svc__C':[1,5,10,50],'svc__gamma':[0.0001,0.0005,0.001,0.005]}

grid=GridSearchCV(model,param_grid)
grid.fit(Xtrain,ytrain)
print(grid.best_params_)
