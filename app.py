from flask import Flask, render_template, request, redirect
import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np
import os

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
	img=request.files['input']
	directory='/home/mehedi/test/project5_clean/static/uploaded_images/'
	files=os.listdir(directory)
	count=len(files)
	filename=str(count)+'_'+img.filename
	path=directory+filename
	img.save(path)

	img=cv.imread(path)
	img_res=cv.resize(img,(128,58))
	feature=img_res.reshape(1,img_res.shape[0],img_res.shape[1],img_res.shape[2])
	feature_norm=feature/255.0

	model_path='/home/mehedi/test/project5_clean/cnn_model.h5'
	cnn=load_model(model_path)
	results=cnn.predict(feature_norm)
	y_hat=np.argmax(results)

	labels=['2', '5', '10', '20', '50', '100', '500', '1000']

	return render_template("index.html", dbg=results.shape, prediction=labels[y_hat], img_path=filename)

if __name__=='__main__':
	app.run(debug=True)