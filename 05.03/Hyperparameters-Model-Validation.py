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

# KNN ���趨�Ľ��ڸ���Ϊ1�����Ӧ���ǲ���ȷ�ģ������������Ҫ������

model=KNeighborsClassifier(n_neighbors=1)

# ѵ��ģ��

model.fit(X,y)

# Ԥ���ǩֵ

y_model=model.predict(X)

# ��Ԥ�������д��

from sklearn.metrics import accuracy_score
accuracy_score(y,y_model)

# �����ݴ���� ѵ�����ϲ��Լ�

from sklearn.cross_validation import train_test_split

# ������ƽ�ֳ�ѵ�����ϲ��Լ�

X1,X2,y1,y2=train_test_split(X,y,random_state=0,train_size=0.5)

# ����ѵ����ѵ��ģ��

model.fit(X1,y1)

# Ԥ����Լ��ı�ǩ

y2_model=model.predict(X2)

# ��Ԥ�������д��

accuracy_score(y2,y2_model)

print("#---------------------------------------#")
print("            Cross validation             ")
print("#---------------------------------------#")
print("\n")

# Two-fold cross-validation

y2_model=model.fit(X1,y1).predict(X2)
y1_model=model.fit(X2,y2).predict(X1)
accuracy_score(y1,y1_model),accuracy_score(y2,y2_model)

# ���� cross_validation �е� cross_val_score �������Ժܺõ�ʵ��
# n-fold cross-validation ����

from sklearn.cross_validation import cross_val_score

# ���������һ������ model,˵������������Ҫѡ�����Ҫ��ģ��ֵ
cross_val_score(model,X,y,cv=5)

# �� cross_validation ģ���б�����һϵ�еĽ��н�����֤�Ĳ���

# �����ǽ� folds �ĸ����ȼ����� data points

from sklearn.cross_validation import LeaveOneOut
scores=cross_val_score(model,X,y,cv=LeaveOneOut(len(X)))

# �� scores ��ƽ�����ܵõ���ȷ��

scores.mean()
