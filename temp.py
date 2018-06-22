from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

digits=datasets.load_digits()

images_and_labels = list(zip(digits['images'], digits['target']))


for index,(image,label) in enumerate (images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Training: %i' %label)

n_samples=len(digits['images'])

data=digits['images'].reshape(n_samples,-1)  #converting 8X8 to 64X1

#create a classifier SVM

clf=SVC(gamma=0.001)
clf.fit(data[n_samples //2:],digits['target'][n_samples //2 :])

expected=digits['target'][: n_samples //2]
predicted=clf.predict(data[: n_samples //2])


# calculating mispredictions using metrics

from sklearn import metrics
y=metrics.confusion_matrix(expected,predicted)

image_and_prediction=list(zip(digits['images'][: n_samples//2],predicted))

for index,(image,prediction) in enumerate (image_and_prediction[:6]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('prediction : %i' % prediction)
    

plt.show()    


# importing my own data into our model

from scipy import misc  #Read an image from a file as an array.

img1=misc.imread(r'C:\Users\R558UF\Desktop\M_L project1\8.jpg')
img1=misc.imresize(img1,(8,8))
img1=img.astype(digits['images'].dtype)
img=misc.bytescale(img,high=16,low=0)

x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)

        
x_final=np.reshape(x_test,(8,8))

print(clf.predict([x_test]))

plt.imshow(x_final,cmap= plt.cm.gray_r,interpolation='nearest')
plt.axis('off')






    
    
    







   

    
    
    












