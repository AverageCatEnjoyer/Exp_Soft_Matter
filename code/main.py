import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import cv2

# Example NumPy arrays
# frames = np.array([1, 1, 2, 2, 3, 3])
# particles = np.array([1, 2, 1, 2, 1, 2])
# x_coords = np.array([10.2, 15.5, 11.3, 16.1, 12.7, 17.2])
# y_coords = np.array([5.5, 8.1, 6.2, 9.0, 7.3, 9.5])
#
# # Create a DataFrame with the particle data
# particle_data = pd.DataFrame({'frame': frames, 'particle': particles, 'x': x_coords, 'y': y_coords})
# print(particle_data)
# # Set the maximum displacement for linking particles
# max_displacement = 5.0  # Adjust this value based on your data
#
# # Use the tp.link function to link particles across frames
# linked_particles = tp.link(particle_data, search_range=max_displacement)
# tp.plot_traj(linked_particles)
#
# exit()

output_path = 'D:/Exp_Soft_Matter/results/'

def read_positions_from_txt(filename):
    FRAMES = []
    with open(filename, 'r') as file:
        for line in file:
            shape_info = list(map(int, line.strip().split(',')))
            flattened_frame = np.loadtxt(file, dtype=int, max_rows=np.prod(shape_info))
            frame = flattened_frame.reshape(shape_info)
            FRAMES.append(frame)
    return FRAMES

FRAMES = read_positions_from_txt(output_path+'positions.txt')
for frame in FRAMES:
    print(frame)
exit()
# maximum displacement to link particles
max_displacement = 5.0


# link particles across frames (memory = number of frames for a particle to disappear before being excluded)
linked_particles = tp.link_df(FRAMES, search_range=max_displacement, memory=1)

# Filter tracks based on minimum length
# min_track_length = 3
# filtered_tracks = tp.filter_stubs(linked_particles, min_track_length)

tp.plot_traj(linked_particles)

# # Example usage
# for frame in FRAMES:
#
#
#
#
#
exit()










# load the image
image = cv2.imread('D:/Exp_Soft_Matter/images/Exercise0000.jpg')
gray = image[:,:,0]

# blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# detect circles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=5,
    param1=50,
    param2=10,
    minRadius=4,
    maxRadius=9
)

# draw circles into image
if circles is not None:
    print(f'number of circles: {len(circles[0,:])}')
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw outer circle
        # cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw center
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    print(f'number of circles: 0')


# resize image
display_width = 1000
display_height = 1000
resized_image = cv2.resize(image, (display_width, display_height))

cv2.imshow('Circles', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
