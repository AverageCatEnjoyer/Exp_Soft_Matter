import cv2
import numpy as np
from tqdm import tqdm
import os

FRAMES=[]
VIDEO = cv2.VideoCapture('Exercise.avi')

# Check if video was opened
if not VIDEO.isOpened():
    print("\n\n-----------------------------------------")
    print("    Error: Could not open video file.")
    print("-----------------------------------------\n\n")
    exit()

total_frames = int(VIDEO.get(cv2.CAP_PROP_FRAME_COUNT))

# Read frame by frame
# for frame_idx in tqdm(range(total_frames)):
for frame_idx in tqdm(range(5)):
    # Read single frame
    ret, frame = VIDEO.read()


    FRAMES.append(frame)
    # cv2.imshow('Video Frame', frame)

    #manipulate code here

    # Break the loop by pressing 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# combine manipulated frames to video:

# get dimensions
height, width, layers = FRAMES[0].shape

print()
# Create a VideoWriter object
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Loop through each frame and write it to the video
for frame_file in frame_files:
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    # Write the frame to the video
    video_writer.write(frame)



# Release the video capture object and close the window
VIDEO.release()
cv2.destroyAllWindows()
