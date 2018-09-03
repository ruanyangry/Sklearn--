# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

Feature Extraction Features: Histogram of Oriented Gradients
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

from skimage import data,color,feature
import skimage.data

image=color.rgb2gray(data.chelsea())
hog_vec,hog_vis=feature.hog(image,visualise=True)

print("#-------------------------------------------#")
print(hog_vec)
print(hog_vec.size)
print("#-------------------------------------------#")
print("\n")

fig,ax=plt.subplots(1,2,figsize=(12,6),subplot_kw=dict(xticks=[],\
yticks=[]))

ax[0].imshow(image,cmap="gray")
ax[0].set_title("Input Image")

ax[1].imshow(hog_vis)
ax[1].set_title("Visualization of HOG features")

plt.savefig("5.14-1.jpg",dpi=300)
plt.show()

# Build a simple facial detection algorithm
# Using linear SVM
# Steps
# 1. Obtain a set of image thumbnails of faces to constitute \
# "positive" training samples.
# 2. Obtain a set of image thumbnails of non-faces to constitute \
# "negative" training samples.
# 3. Extract HOG features from these training samples.
# 4. Train a linear SVM classifier on these samples.
# 5. For an "unknown" image, pass a sliding window across the image, \
# using the model to evaluate whether that window contains a face or not.
# 6. If detections overlap, combine them into a single window.

# 1. Obtain a set of positive training samples

from sklearn.datasets import fetch_lfw_people
faces=fetch_lfw_people()
positive_patches=faces.images

print("#-------------------------------------------#")
positive_patches.shape
print("#-------------------------------------------#")
print("\n")

# 2. Obtain a set of negative training samples

from skimage import data,transform

imgs_to_use=['camera', 'text', 'coins', 'moon',\
'page', 'clock', 'immunohistochemistry',\
'chelsea', 'coffee', 'hubble_deep_field']

images=[color.rgb2gray(getattr(data,name)()) for name in imgs_to_use]

from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img,N,scale=1.0,patch_size=positive_patches[0].shape):
	extract_patches_size=tuple((scale*np.array(patch_size)).astype(int))
	extractor=PatchExtractor(patch_size=extract_patches_size,max_patches=N,\
	random_state=0)
	patches=extractor.transform(img[np.newaxis])
	if scale != 1:
		patches=np.array([transform.resize(patch,patch_size) for patch in patches])
	return patches
	
negative_patches=np.vstack([extract_patches(im,1000,scale) for im in images \
for scale in [0.5,1.0,2.0]])

print("#-------------------------------------------#")
negative_patches.shape
print("#-------------------------------------------#")
print("\n")

fig,ax=plt.subplots(6,10)

for i,axi in enumerate(ax.flat):
	axi.imshow(negative_patches[500*i],cmap="gray")
	axi.axis("off")
	
plt.savefig("5.14-2.jpg",dpi=300)
plt.show()
	
# 3. Combine sets and extract HOG

from itertools import chain

X_train=np.array([feature.hog(im) for im in chain(positive_patches,\
negative_patches)])

# Get the training data

y_train=np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]]=1

print(X_train.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score

cross_val_score(GaussianNB(),X_train,y_train)

# Using Grid Search methods find the hyperparameter C

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

grid=GridSearchCV(LinearSVC(),{'C':[1.0,2.0,4.0,8.0]})
grid.fit(X_train,y_train)

print("#-------------------------------------------#")
grid.best_score_
grid.best_params_
print("#-------------------------------------------#")
print("\n")

model=grid.best_estimator_
model.fit(X_train,y_train)

# 5. Find faces in a new image

# Get the new image

test_image=skimage.data.astronaut()

# translate RGB to Gray
test_image=skimage.color.rgb2gray(test_image)
test_image=skimage.transform.rescale(test_image,0.5)
test_image=test_image[:160,40:180]

plt.imshow(test_image,cmap="gray")
plt.axis("off")

plt.savefig("5.14-3.jpg",dpi=300)
plt.show()

# create a window slide

def sliding_window(img,patch_size=positive_patches[0].shape,istep=2,\
jstep=2,scale=1.0):
	Ni,Nj=(int(scale*s) for s in patch_size)
	for i in range(0,img.shape[0]-Ni,istep):
		for j in range(0,img.shape[1]-Ni,jstep):
			patch=img[i:i+Ni,j:j+Nj]    # defined a subset zone in image
			if scale != 1:
				patch=transform.resize(patch,patch_size)
			yield (i,j),patch
			
indices,patches=zip(*sliding_window(test_image))
patches_hog=np.array([feature.hog(patch) for patch in patches])

print("#-------------------------------------------#")
patches_hog.shape
print("#-------------------------------------------#")
print("\n")

labels=model.predict(patches_hog)

print("#-------------------------------------------#")
labels.sum()
print("#-------------------------------------------#")
print("\n")

# Show these patches in our test image, drawing them as rectangles

fig,ax=plt.subplots()
ax.imshow(test_image,cmap="gray")
ax.axis("off")

Ni,Nj=positive_patches[0].shape
indices=np.array(indices)

for i,j in indices[labels == 1]:
	ax.add_patch(plt.Rectangle((j,i),Nj,Ni,edgecolor='red',alpha=0.3,\
	lw=2,facecolor="none"))

plt.savefig("5.14-4.jpg",dpi=300)
plt.show()
