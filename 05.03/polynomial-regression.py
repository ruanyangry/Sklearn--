import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("         Polynomial regression           ")
print("#---------------------------------------#")
print("\n")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2,**kwargs):
	return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))
	
def make_data(N,err=1.0,rseed=1):
	# randomly sample the data
	rng=np.random.RandomState(rseed)
	X=rng.rand(N,1)**2
	y=10-1./(X.ravel()+0.1)
	if err > 0 :
		y += err*rng.randn(N)
	return X,y
	
# X: feature matrix
# y: target vector

X,y=make_data(40)

import seaborn
seaborn.set()

X_test=np.linspace(-0.1,1.1,500)[:,None]

plt.scatter(X.ravel(),y,color='black')
axis=plt.axis()
for degree in [1,3,5]:
	y_test=PolynomialRegression(degree).fit(X,y).predict(X_test)
	plt.plot(X_test.ravel(),y_test,label="degree=%d"%(degree))
plt.xlim(-0.1,1.0)
plt.ylim(-2,12)
plt.legend(loc='best')
plt.show()

from sklearn.learning_curve import validation_curve
# The max degree = 20
degree=np.arange(0,21)

train_score,val_score=validation_curve(PolynomialRegression(),X,y,'polynomialfeatures_degree',\
degree,cv=7)

plt.plot(degree,np.median(train_score,1),color='blue',label='training score')
plt.plot(degree,np.median(val_score,1),color='red',label='validation value')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

	
