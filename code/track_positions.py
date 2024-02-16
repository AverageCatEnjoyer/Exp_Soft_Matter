import cv2
import trackpy as tp
import numpy as np
from tqdm import tqdm
import os

# uni
# frames_directory = '/local/mroot/Exp_Soft_Matter/images'
# output_path = '/local/mroot/Exp_Soft_Matter/results/video.mp4'

# home
frames_directory = 'D:/Exp_Soft_Matter/images'
output_path = 'D:/Exp_Soft_Matter/results/'


def write_positions_to_txt(frames, filename):
    with open(filename, 'w') as file:
        for frame in frames:
            shape_info = np.array(frame.shape)
            flattened_frame = frame.flatten()
            file.write(','.join(map(str, shape_info)) + '\n')
            np.savetxt(file, flattened_frame, fmt='%d')


POSITIONS=[]


# list of frame files
FRAME_NAMES = sorted([f for f in os.listdir(frames_directory) ])
total_frames = len(FRAME_NAMES)


# get dimensions
first_frame = cv2.imread(os.path.join(frames_directory, FRAME_NAMES[0]))
height, width, layers = first_frame.shape



# Create a VideoWriter object with 30 fps
video_writer = cv2.VideoWriter(output_path+"video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

# Loop through each frame and write it to the video
for frame_number,frame_name in enumerate(tqdm(FRAME_NAMES)):
    DUMMY=[]

    # choose current frame
    frame_path = os.path.join(frames_directory, frame_name)
    frame = cv2.imread(frame_path)

    # reduce to 1 layer only
    gray_frame = frame[:,:,0]

    # apply gaussian blur to reduce noise and threshhold
    blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    # detect particles
    circles = cv2.HoughCircles(
        blurred_frame,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=5,
        param1=50,
        param2=10,
        minRadius=4,
        maxRadius=9
    )

    # # detect holes
    # circles = cv2.HoughCircles(
    #     blurred_frame,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1,
    #     minDist=50,
    #     param1=50,
    #     param2=10,
    #     minRadius=35, #202 for middle
    #     maxRadius=35  #202 for middle
    # )

    # draw circles into image
    if circles is not None:
        # print(f'number of circles: {len(circles[0,:])}')

        circles = np.uint16(np.around(circles))
        for circle_number,i in enumerate(circles[0, :]):
            DUMMY.append((frame_number, circle_number, i[0], i[1])) # (frame_number,particle_number,x,y)

            # draw outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # draw center
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print(f'number of circles: 0')

    #-----------------------------------------------------------------------------------
    # # resize image
    # display_width = 1000
    # display_height = 1000
    # resized_frame = cv2.resize(frame, (display_width, display_height))
    # cv2.imshow('Frame', resized_frame)
    # cv2.waitKey(0)
    # exit()
    #-----------------------------------------------------------------------------------


    # save circle positions into array
    POSITIONS.append(np.array(DUMMY))

    # print(POSITIONS)

    # Write the frame to the video
    video_writer.write(frame)


# save positions in chosen shape
write_positions_to_txt(POSITIONS, output_path+'positions.txt')

# Release videowriter object
video_writer.release()
