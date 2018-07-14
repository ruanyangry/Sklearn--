# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("       simple linear regression          ")
print("#---------------------------------------#")
print("\n")

rng=np.random.RandomState(42)
x=10*rng.rand(50)
y=2*x-1+rng.rand(50)
plt.scatter(x,y)
plt.show()

# import sklearn

from sklearn.linear_model import LinearRegression

model=LinearRegression(fit_intercept=True)

X=x[:,np.newaxis]

model.fit(X,y)

print("Fit line coefficient = %.4f"%(model.coef_))
print("Intercept = %.8f"%(model.intercept_))

xfit=np.linspace(-1,11)

Xfit=xfit[:,np.newaxis]
yfit=model.predict(Xfit)

plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()

print("#---------------------------------------#")
print("          Iris classification            ")
print("#---------------------------------------#")
print("\n")

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X_iris,y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(Xtrain,ytrain)
y_model=model.predict(Xtest)

# accuracy_score() 应该是对模型的预测结果进行打分
from sklearn.metrics import accuracy_score
accuracy_score(ytest,y_model)

print("#---------------------------------------#")
print("                    PCA                  ")
print("#---------------------------------------#")
print("\n")

from sklearn.decomposition import PCA
model=PCA(n_components=2)   # 降数据降到2维
model.fit(X_iris)
X_2D=model.transform(X_iris)

iris['PCA1']=X_2D[:,0]
iris['PCA2']=X_2D[:,1]
sns.lmplot("PCA1","PCA2",hue='species',data=iris,fit_reg=False)

print("#---------------------------------------#")
print("               Clustering                ")
print("#---------------------------------------#")
print("\n")

# Gaussian mixture model (GMM)
# covariance:协方差

from sklearn.mixture import GMM
model=GMM(n_components=3,covariance_type='full')
model.fit(X_iris)
y_gmm=model.predict(X_iris)

iris['cluster']=y_gmm
sns.lmplot("PCA1","PCA2",data=iris,hue='species',col='cluster',fit_reg=False)


print("#---------------------------------------#")
print("           Hand-written digits           ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_digits

digits=load_digits()
digits.images.shape

plt.clf()

fig,axes=plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},\
gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i,ax in enumerate(axes.flat):
	ax.imshow(digits.images[i],cmap="binary",interpolation='nearest')
	ax.text(0.05,0.05,str(digits.target[i]),transform=ax.transAxes,color='green')
	
X=digits.data
X.shape

y=digits.target
y.shape

# Manifold learning 降维操作

from sklearn.manifold import Isomap

iso=Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

plt.scatter(data_projected[:,0],data_projected[:,1],c=digits.target,\
edgecolor='none',alpha=0.5,cmap=plt.cm.get_cmap('spectral',10))
plt.colorbar(label='digit label',ticks=range(10))
plt.clim(-0.5,9.5)

print("#---------------------------------------#")
print("       Classification on digits          ")
print("#---------------------------------------#")
print("\n")

# 拆分数据集

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0)

from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(Xtrain,ytrain)
y_model=model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,y_model)

from sklearn.metrics import confusion_matrix

mat=confusion_matrix(ytest,y_model)

sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

