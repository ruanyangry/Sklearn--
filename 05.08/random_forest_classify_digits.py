# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

print("#---------------------------------#")
print('          classifying digits       ')
print("#---------------------------------#")
print("\n")

from sklearn.datasets import load_digits
digits=load_digits()
digits.keys()
print(digits.keys())

fig=plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

for i in range(64):
	ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
	ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
	ax.text(0,7,str(digits.target[i]))
	
plt.show()

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
Xtrain,Xtest,ytrain,ytest=train_test_split(digits.data,digits.target,random_state=0)
model=RandomForestClassifier(n_estimators=100)
model.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)

from sklearn import metrics
print(metrics.classification_report(ypred,ytest))

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(ytest,ypred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()
