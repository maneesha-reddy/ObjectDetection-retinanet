from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
from keras import backend
#from tensorflow.keras import backend

#tf.compat.v1.Session()
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#self.sess = tf.compat.v1.Session()
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "/home/ubuntu/ciphence/static/images/123.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))


detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "/home/ubuntu/ciphence/static/images/123.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)
print(extracted_images)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )