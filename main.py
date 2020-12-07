import cv2
import numpy as np
import pandas as pd
import os
import sys
import math
import itertools

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names

#Kui kasutate vanemat (v1) Tensorflow versiooni, 
#siis äkki peab siin midagi muutma, kui ei tööta
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_v2_behavior() 

import tensorflow as tf
from scipy.spatial import distance as dist
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm_notebook as tqdm


import time
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

from pathlib import Path

def video_to_images(video, size = None):
	# Params: video as VideoCapture
	# Yields resized images
	# video = cv2.VideoCapture(video_filename)
	success, image = video.read()
	
	while success:
		# resize image
		
		yield image
		success, image = video.read()
		
	video.release();
	
def calculate_coord(bbox, width, height):
	"""Return boxes coordinates"""
	
	xmin = bbox[1] * width
	ymin = bbox[0] * height
	xmax = bbox[3] * width
	ymax = bbox[2] * height

	return [xmin, ymin, xmax - xmin, ymax - ymin]
	
def calculate_centr(coord):
	"""Calculate centroid for each box"""
	return (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
  

def calculate_centr_distances(centroid_1, centroid_2):
	
	"""Calculate the distance between 2 centroids"""
	return math.sqrt((centroid_2[0]-centroid_1[0])**2 + (centroid_2[1]-centroid_1[1])**2)
	
def calculate_perm(centroids):
	"""Return all combinations of centroids"""
	permutations = []
	for current_permutation in itertools.permutations(centroids, 2):
		if current_permutation[::-1] not in permutations:
			permutations.append(current_permutation)
	return permutations
  
def calc_midpoint(p1, p2):
	"""Midpoint between 2 points"""
	return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

## TODO - fill this with something

def calculate_distances(permutations, rects, width, height):
	# Permutations - list of tuples [((x1, y1),(x2, y2))]
	# Rects - Lists of  [x, y, width, height]
	# Size - (width, height)
	distances = []
	
	for perm in permutations:
		#first centroid  coordinate
		x1 = perm[0][0]
		y1 = perm[0][1]
		#second centroid coorinate
		x2 = perm[1][0]
		y2 = perm[1][1]
		
		average_px_meter = (width-540) / 10 #lambist proovitud arv
		
		dist = calculate_centr_distances(perm[0], perm[1])
		#print(dist)
		dist_m = dist/ average_px_meter
		#print("M meters: ", str(dist_m))

		distances.append(dist_m)
	# Returns distances - list of distances as numbers
	
	return distances
	
def draw_rects(image, coordinates, color = (0, 0, 255), thickness = 3):
	# Draws rectangles onto the image
	# input list of Lists of  [x, y, width, height] 
	# color is tuple in BGR
	# thickness is thickness of line in pixels
	
	for i in range(len(coordinates)):
		coord = coordinates[i]

		x1 = int(coord[0])
		y1 = int(coord[1])
		x2 = x1 + int(coord[2])
		y2 = y1 + int(coord[3])

		image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
	
	return image

def draw_lines(image, permutations, distances, color = (0, 255, 255), thickness = 1):
	# Given pairs of permutations
	# Draws lines between Centroids 
	
	for i, perm in enumerate(permutations):
		point1, point2 = perm[0], perm[1]
		point1 = tuple(map(int, point1))
		point2 = tuple(map(int, point2))
		x1 = point1[0]
		y1 = point1[1]
		x2 = point2[0]
		y2 = point2[1]
		image = cv2.line(image, point1, point2, color, thickness)
		mid = tuple(map(int, calc_midpoint(point1, point2)))
 
		if round(distances[i], 1) < 2.0:
			cv2.putText(image, str(round(distances[i], 1)) +'m', mid, font, 2, (0, 0, 255), 2, cv2.LINE_AA)
		else:
			cv2.putText(image, str(round(distances[i], 1)) +'m', mid, font, 2, (255, 255, 255), 2, cv2.LINE_AA)
			
		
	return image
	
input_video_filename = 'data/test_video_4K_25fps.mp4'
output_video_filename = 'data/output.avi'

font = cv2.FONT_HERSHEY_SIMPLEX

input_video_filepath = input_video_filename
output_video_filepath = output_video_filename
print("Input  - ",input_video_filepath)
print("Output - ",output_video_filepath)

input_video = cv2.VideoCapture(input_video_filepath)
fps = input_video.get(cv2.CAP_PROP_FPS)

Width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
Height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_filepath, fourcc, fps, (Width, Height))

times = []

CLASSES  = "model_data/coco.names"
NUM_CLASS = read_class_names(CLASSES)
key_list = list(NUM_CLASS.keys()) 
val_list = list(NUM_CLASS.values())

classes = None
with open('coco.names.txt', 'r') as f:
	classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('./model_data/yolov3.weights', 'yolov3.cfg') # or ./model_data/yolov4.weights and yolov4.cfg
max_cosine_distance = 0.7
nn_budget = None

			#initialize deep sort object
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

try: 
	with tqdm(total=total_frames) as pbar:
		for image in video_to_images(input_video):
			t1 = time.time()
			net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

			layer_names = net.getLayerNames()
			output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
			outs = net.forward(output_layers)


			class_ids = []
			confidences_l = []
			boxes_l = []
			names_l = []
			#create bounding box 
			
			for out in outs:
				for detection in out:
					
					scores = detection[5:]
					class_id = np.argmax(scores)
					if class_id == 0:
						confidence = scores[class_id]
						if confidence > 0.1:
							center_x = int(detection[0] * Width)
							center_y = int(detection[1] * Height)
							w = int(detection[2] * Width)
							h = int(detection[3] * Height)
							x = int((center_x - w / 2))
							y = int((center_y - h / 2))
							class_ids.append(class_id)
							confidences_l.append(float(confidence))
							boxes_l.append([x, y, w, h])
							names_l.append(classes[class_id])
							#print(names)
						   
			boxes = np.array(boxes_l) 
			names = np.array(names_l)
			scores = np.array(confidences_l)
			
			features = np.array(encoder(image, boxes))
			
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
			
			# Pass detections to the deepsort object and obtain the track information.
			tracker.predict()
			tracker.update(detections)
			
			tracked_bboxes = []
			for track in tracker.tracks:
				#print(track)
				
				if not track.is_confirmed() or track.time_since_update > 5:
					continue
				bbox = track.to_tlbr() # Get the corrected/predicted bounding box
		
				class_name = track.get_class() #Get the class name of particular object
				tracking_id = track.track_id # Get the ID for the particular track
				index = key_list[val_list.index(class_name)] # Get predicted object index by object name
				
				tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

			
			# draw detection on frame
			image = draw_bbox(image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
		
			indices = cv2.dnn.NMSBoxes(boxes_l, confidences_l, 0.1, 0.1)

			# Tuples of (x, y) coords
			centroids = []
			# Lists of  [x, y, width, height]
			coordinates = []
			for i in indices:
				i = i[0]
				box = boxes[i]
				centr = calculate_centr(box)
				centroids.append(centr)
				coordinates.append(box)
				#if class_ids[i]==0:
				#	label = str(classes[class_id]) 
				#	cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
					#cv2.putText(image,(round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		   
		
			# Calculate all possible connections between centroids
			permutations = calculate_perm(centroids)
			distances = calculate_distances(permutations, coordinates, Width, Height)
		
			#draw_rects(image, coordinates)
			draw_lines(image, permutations, distances)
			# Write to output video file
			output_video.write(image)
			pbar.update(1)
			t3 = time.time()
			
			times.append(t3-t1)
			
			times = times[-20:]
			

			ms = sum(times)/len(times)*1000
			
			print("Time: {:.2f}ms".format(ms))
except KeyboardInterrupt:
	print("Process stopped")
finally:
	print("Output saved")
	output_video.release() 

