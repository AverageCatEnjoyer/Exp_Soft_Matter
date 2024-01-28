import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import cv2
from tqdm import tqdm

output_path = 'D:/Exp_Soft_Matter/results/'

# useful lists
FRAMES=[]
PARTICLES=[]
X_COORDS=[]
Y_COORDS=[]


# function reads txt file of particle positions and creates lists used for tracking
def read_positions_from_txt(filename):
    with open(filename, 'r') as FILE:
        for line in tqdm(FILE):
            shape_info = list(map(int, line.strip().split(',')))
            flattened_frame = np.loadtxt(FILE, dtype=int, max_rows=np.prod(shape_info))
            FRAME = flattened_frame.reshape(shape_info)
            for data in FRAME:
                FRAMES.append(data[0])
                PARTICLES.append(data[1])
                X_COORDS.append(data[2])
                Y_COORDS.append(data[3])



read_positions_from_txt(output_path+'holes.txt')


# create particle dataframe
PARTICLE_DATA = pd.DataFrame({'frame': FRAMES, 'particle': PARTICLES, 'x': X_COORDS, 'y': Y_COORDS})
# print(PARTICLE_DATA)


# maximum displacement to link particles
max_displacement = 5

# link particles across frames (memory = number of frames for a particle to disappear before being excluded)
linked_particles = tp.link_df(PARTICLE_DATA, search_range=max_displacement, memory=1)

# Filter tracks based on minimum length
min_track_length = 3
filtered_tracks = tp.filter_stubs(linked_particles, min_track_length)

tp.plot_traj(filtered_tracks)

exit()
