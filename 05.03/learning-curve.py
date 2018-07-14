# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("             learning curve              ")
print("#---------------------------------------#")
print("\n")

from sklearn.learning_curve import learning_curve

fig,ax=plt.subplots(1,2,figsize=(16,6))
fig.subplots_adjust(left=0.0625,right=0.95,wspace=0.1)

for i,degree in enumerate([2,9]):
	N,train_lc,val_lc=learning_curve()
	pass
