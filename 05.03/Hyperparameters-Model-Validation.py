# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("         K-neighbors classifier          ")
print("#---------------------------------------#")
print("\n")

from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data      # feature matrix
y=iris.target    # target vector

from sklearn.neighbors import KNeighborsClassifier

# KNN 中设定的近邻个数为1，这个应该是不正确的，这个参数是需要调整的

model=KNeighborsClassifier(n_neighbors=1)

# 训练模型

model.fit(X,y)

# 预测标签值

y_model=model.predict(X)

# 对预测结果进行打分

from sklearn.metrics import accuracy_score
accuracy_score(y,y_model)

# 将数据处理成 训练集合测试集

from sklearn.cross_validation import train_test_split

# 将数据平分成训练集合测试集

X1,X2,y1,y2=train_test_split(X,y,random_state=0,train_size=0.5)

# 基于训练集训练模型

model.fit(X1,y1)

# 预测测试集的标签

y2_model=model.predict(X2)

# 对预测结果进行打分

accuracy_score(y2,y2_model)

print("#---------------------------------------#")
print("            Cross validation             ")
print("#---------------------------------------#")
print("\n")

# Two-fold cross-validation

y2_model=model.fit(X1,y1).predict(X2)
y1_model=model.fit(X2,y2).predict(X1)
accuracy_score(y1,y1_model),accuracy_score(y2,y2_model)

# 基于 cross_validation 中的 cross_val_score 方法可以很好的实现
# n-fold cross-validation 操作

from sklearn.cross_validation import cross_val_score

# 这里面存在一个变量 model,说明我们事先需要选择好需要的模型值
cross_val_score(model,X,y,cv=5)

# 在 cross_validation 模块中保存了一系列的进行交叉验证的策略

# 下例是将 folds 的个数等价于了 data points

from sklearn.cross_validation import LeaveOneOut
scores=cross_val_score(model,X,y,cv=LeaveOneOut(len(X)))

# 将 scores 求平均就能得到正确率

scores.mean()
