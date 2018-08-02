"""creating a web app or hand written digit recognizing  machine learning model """
from flask import Flask, render_template,url_for, request, jsonify
from PIL import Image
from scipy import misc
import re
import io
import base64


# machine learning libraries

from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
    
    
app=Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/',methods=['GET' ,'POST'])
def get_image():
    guess=0
    
    
    if request.method=='POST':
        digits=datasets.load_digits()
        n_samples=len(digits['images'])
        data=digits['images'].reshape(n_samples,-1)  #converting 8X8 to 64X1
        #create our SVM classifier model
        clf=SVC(gamma=0.001)
        clf.fit(data,digits['target'])
        img_size=8,8
        image_url=request.values['imageBase64']
        image_string=re.search(r'base64,(.*)',image_url).group(1)
        image_bytes=io.BytesIO(base64.b64decode(image_string)) #image_bytes is the address
        image=misc.imread(image_bytes)
        image=misc.imresize(image,img_size)
        image=image.astype(digits['images'].dtype)
        image=misc.bytescale(image,high=16,low=0)
        x_test = []
        for eachRow in image:
            for eachPixel in eachRow:
                x_test.append(sum(eachPixel)/4.0)
        print("hello world")        
        print(x_test)        
        guess=clf.predict(x_test)
        print("i am guess")
        print(guess)
        
    return render_template('results.html',guess=guess)


if __name__=='__main__':
    app.run(debug=True)