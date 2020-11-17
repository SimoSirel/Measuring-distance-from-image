import cv2
from detection_utils import *

## TODO - fill this with something
def calculate_distances(permutations, rects, width, height):
    # Permutations - list of tuples [((x1, y1),(x2, y2))]
    # Rects - Lists of  [x, y, width, height]
    # Size of image - (width, height)
    #for perm in permutations:
    #    x1 = perm[0][0]
    #    x2 = perm[1][0]
    #    y1 = perm[0][1]
    #    y2 = perm[1][1]
    #
    # calculate_centr_distances(point1, point2)  for distance between 2 points
    # Returns distances - list of distances as numbers
    distances = []
    for i in range(len(permutations)):
        distances.append(i/2.1) 
    
    return distances

def draw_rects(image, coordinates, color = (0, 0, 255), thickness = 5):
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

def draw_lines(image, permutations, distances, color = (0, 255, 255), thickness = 3, font = cv2.FONT_HERSHEY_SIMPLEX):
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
        
        #TODO - write distance on line
        mid = tuple(map(int, calc_midpoint(point1, point2)))
        # Font from beginning of main
        # image, text, point, font, 
        cv2.putText(image, str(round(distances[i], 2)), mid, font, 3, (0, 0, 0), 4, cv2.LINE_AA)
        
    return image