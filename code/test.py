import cv2
import numpy as np
from tqdm import tqdm
import os


FRAMES=[]

frames_directory = '/local/mroot/Exp_Soft_Matter/images'

output_video_path = '/local/mroot/Exp_Soft_Matter/results/video.mp4'

# list of frame files 
FRAME_NAMES = sorted([f for f in os.listdir(frames_directory) ])

total_frames = len(FRAME_NAMES)

# !!!check at home if order is correct!!!
# print(FRAMES[0])
# print(FRAMES[10])
# exit()


# get dimensions
first_frame = cv2.imread(os.path.join(frames_directory, FRAME_NAMES[0]))
height, width, layers = first_frame.shape
# print(height,width,layers)

#------------------------------------------------------------------------------------
#testing

frame_path = os.path.join(frames_directory, FRAME_NAMES[0])
frame = cv2.imread(frame_path)


gray_frame = frame[:,:,0] 

blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
_, thresholded_frame = cv2.threshold(blurred_frame, 145, 255, cv2.THRESH_BINARY)
cv2.imshow('Original Grayscale Frame', gray_frame)
cv2.imshow('Thresholded Frame', thresholded_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
#------------------------------------------------------------------------------------

# Create a VideoWriter object with 30 fps
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=False)

# Loop through each frame and write it to the video
for frame_name in tqdm(FRAME_NAMES):
    frame_path = os.path.join(frames_directory, frame_name)
    frame = cv2.imread(frame_path)

    gray_frame = frame[:,:,0]

    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    thresholded_frame = cv2.threshold(gray_frame, 130, 255, cv2.THRESH_BINARY)

    cv2.imshow('Original Grayscale Frame', gray_frame)
    cv2.imshow('Thresholded Frame', thresholded_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()


    # Write the frame to the video
    video_writer.write(gray_frame)



# Release videowriter object
video_writer.release()
# cv2.destroyAllWindows()
