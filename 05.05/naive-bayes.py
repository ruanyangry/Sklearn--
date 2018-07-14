# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('#--------------------------------#')
print('       Gaussian Naive Bayes       ')
print('#--------------------------------#')
print("\n")

from sklearn.datasets import make_blobs
X,y=make_blobs(100,2,centers=2,random_state=2,cluster_std=1.5)
print(y)
print("\n")
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="RdBu")
plt.show()

# import GaussianNB in naive_bayes

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()  # get the model
model.fit(X,y)      # fit data to get the coefficient

# Generated new data
rng=np.random.RandomState(0)
Xnew=[-6,-14]+[14,18]*rng.rand(2000,2)
ynew=model.predict(Xnew)    # Using model to predict

# Draw new plot contain raw and new data

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='RdBu')
lim=plt.axis()
plt.scatter(Xnew[:,0],Xnew[:,1],c=ynew,s=20,cmap='RdBu',alpha=0.5)
plt.axis(lim)
plt.show()

yprob=model.predict_proba(Xnew)
print(yprob[-8:].round(2))
print("\n")

print('#--------------------------------#')
print('     Multinomial Naive Bayes      ')
print('#--------------------------------#')
print("\n")

# Get the database through internet

from sklearn.datasets import fetch_20newsgroups

data=fetch_20newsgroups()
print(data.target_names)   

# set the train and test subset of data.

categories = ['talk.religion.misc', 'soc.religion.christian',
'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(),MultinomialNB())

# Train model

model.fit(train.data,train.target)

# Predict value

labels=model.predict(test.data)

# Confusion matrix is very important
# Can used to verify the relationship between predict and target

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,\
xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()
