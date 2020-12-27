import numpy as np
import os
import sys
import cv2

from detection_utils import *
from distance_measuring import *
from tqdm import tqdm

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Example usage: 
    params = {}
    arg_iter = iter(sys.argv)
    # python file name
    next(arg_iter)
    for el in arg_iter:
        params[el] = next(arg_iter)
        
    # If Detection threshold not specified, sets it to 0.3
    confidence_cutoff = float(params.get('-thresh', 0.3))
    
    # If input and output are not specified uses these files
    input_image_filepath = params.get('-in', 'data/input.jpg')
    output_image_filepath = params.get('-out', 'data/output.jpg')  
    
    print("Input  - ", input_image_filepath)
    print("Output - ",output_image_filepath)
    
    # SETUP
    # Model Name
    MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME+"/frozen_inference_graph.pb"

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = "models/research/object_detection/data/mscoco_label_map.pbtxt"

    # Load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    
    image = cv2.imread(input_image_filepath)
    
    width  = image.shape[1]
    height = image.shape[0]

    # Actual detection
    output_dict = run_inference_for_single_image(image, detection_graph)
    
    # Filter the person boxes
    boxes, scores, classes = filter_boxes(confidence_cutoff, output_dict['detection_boxes'], 
        output_dict['detection_scores'], 
        output_dict['detection_classes'], [1])
    
    # Tuples of (x, y) coords
    centroids = []
    # Lists of  [x, y, width, height]
    coordinates = []
    for box in boxes:
        coord = calculate_coord(box, width, height)
        centr = calculate_centr(coord)
        centroids.append(centr)
        coordinates.append(coord)
    
    # Calculate all possible connections between centroids
    permutations = calculate_perm(centroids)
    distances = calculate_distances(permutations, coordinates, width, height)
    
    draw_rects(image, coordinates)
    draw_lines(image, permutations, distances)

    cv2.imwrite(output_image_filepath, image)      