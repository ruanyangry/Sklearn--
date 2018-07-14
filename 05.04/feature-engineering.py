#import matplotlib.pyplot as plt
import numpy as np

print("#---------------------------------------#")
print("           Feature Engineering           ")
print("#---------------------------------------#")
print("\n")

data = [
{'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
{'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
{'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
{'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

#"# sklearn ���ṩ�˽��ֵ��������Ĳ���"

from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False,dtype=int)

# fit_transform()�ĺ����ǼȽ��������Ҳ������ת������

print(vec.fit_transform(data))
print(" ")

# �������������

print(vec.get_feature_names())
print(" ")

# �� sparse ϡ�����ó� True

vec=DictVectorizer(sparse=True,dtype=int)
print(vec.fit_transform(data))
print(" ")

sample = ['problem of evil',
'evil queen',
'horizon problem']

# �����ǳ����ı������е���������Ҫ��ʹ�õ��� word count ����������д���

from sklearn.feature_extraction.text import CountVectorizer

# ���ģ��
vec=CountVectorizer()

# ����ģ�Ͷ����ݽ�����Ϻ�ת��
X=vec.fit_transform(sample)
print(X)
print(" ")

# �� sparse matrix ת���� pandas�е�dataframe

import pandas as pd
print(pd.DataFrame(X.toarray(),columns=vec.get_feature_names()))
print(" ")

from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
X=vec.fit_transform(sample)
print(pd.DataFrame(X.toarray(),columns=vec.get_feature_names()))
print(" ")

# ��������
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([4,2,1,3,7])
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
X=x[:,np.newaxis]
model=LinearRegression().fit(X,y)
yfit=model.predict(X)
plt.scatter(x,y)
plt.plot(x,yfit)
plt.show()

# ���Ӷ����������

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=False)
X2=poly.fit_transform(X)
print(X2)

model=LinearRegression().fit(X2,y)
yfit=model.predict(X2)
plt.scatter(x,y)
plt.plot(x,yfit)
