import numpy as np
import os
import sys
import cv2

from detection_utils import *
from distance_measuring import *
from tqdm import tqdm

if __name__ == '__main__':
    params = {}
    arg_iter = iter(sys.argv)
    # python file name
    next(arg_iter)
    for el in arg_iter:
        params[el] = next(arg_iter)

    # If input and output are not specified uses these files
    input_video_filepath = params.get('-in', 'data/test_video.mp4')
    output_video_filepath = params.get('-out', 'data/smaller_video.avi')
    left_video_filepath = params.get('-out', 'data/smaller_left_video.avi')
    right_video_filepath = params.get('-out', 'data/smaller_right_video.avi')
    fps_reduction = params.get('-fps_reduce', 3)
    size_reduction = params.get('-size_reduce', 4)

    # Initialize VideoCapture
    input_video = cv2.VideoCapture(input_video_filepath)

    # Get parameters from video
    fps = int(input_video.get(cv2.CAP_PROP_FPS)//fps_reduction)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)//size_reduction)  # float
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)//size_reduction)  # float
    #total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    input_video.release()

    print("Input  - ", input_video_filepath)
    print("Output - ", output_video_filepath, "(",width, height,")",fps)

    # Initialize output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter(output_video_filepath, fourcc, fps, (width, height))
    left_video = cv2.VideoWriter(left_video_filepath, fourcc, fps, (width//2, height))
    right_video = cv2.VideoWriter(right_video_filepath, fourcc, fps, (width//2, height))

    input_video = cv2.VideoCapture(input_video_filepath)
    frame_number = 0
    while input_video.isOpened():
        ret, frame = input_video.read()
        frame_number += 1
        #print(frame_number)
        if (frame_number%fps_reduction==0) :
            if frame is not None:
                #print("lisan kaadri")
                small_frame = cv2.resize(frame,(width, height))
                left_frame = small_frame[0:height , 0:width//2]
                right_frame = small_frame[0:height , 0:width//2]

                output_video.write(small_frame)
                left_video.write(left_frame)
                right_video.write(right_frame)

            else:
                break

    input_video.release()
    output_video.release()
    left_video.release()
    right_video.release()
