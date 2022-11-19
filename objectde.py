import imageai.Detection
from imageai.Detection import ObjectDetection
import cv2
import os
execution_path=os.getcwd()
detector=ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,'resnet50_coco_best_v2.1.0.h5'))
detector.loadModel()

detections=detector.detectObjectsFromImage(
               input_image=os.path.join(execution_path,'image.jpg'),
                output_image_path=os.path.join(execution_path,'imagenew.jpg'))
