from flask import Flask
from flask import render_template
from bs4 import BeautifulSoup as BS
from urllib.request import urlopen
import re
from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
from keras import backend
import cv2
from matplotlib import pyplot as plt
import shutil
# creates a Flask application, named app
app = Flask(__name__)
with open("templates/index.html", "r") as f:
	html = f.read()
	bs = BS(html, 'html.parser')
	images = bs.find_all('img', {'src':re.compile('.jpg')})
#f.close()
	print(images)
	message=[]
	i=0
	for image in images: 
		#print(image['src']+'\n')
		x="/home/ubuntu/ciphence/"+image['src']
		print(x)
		execution_path = os.getcwd()
		detector = ObjectDetection()
		detector.setModelTypeAsRetinaNet()
		detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
		detector.loadModel()
		y="imagenew"+str(i)+".jpg"
		detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , x), output_image_path=os.path.join(execution_path , y), extract_detected_objects=True)
		print(execution_path)
		dest='/home/ubuntu/ciphence/static/images'
		src='/home/ubuntu/ciphence'
		#destination = shutil.move(y, dest, copy_function = shutil.copytree)
		shutil.move(os.path.join(src, y), os.path.join(dest, y))
		print(extracted_images)
		message.append(y)
		i=i+1
		for eachObject in detections:
			print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    #message = "Hello, World"
    #return render_template('index.html', message=message)
	return render_template('index.html', message=message)

# run the application
if __name__ == "__main__":
    app.run(debug=True)