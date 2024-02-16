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

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

MIDDLE = (1006.23,987.48) #x,y values of middle point

# read_positions_from_txt(output_path+'holes.txt')
read_positions_from_txt(output_path+'holes.txt')


# create particle dataframe
PARTICLE_DATA = pd.DataFrame({'frame': FRAMES, 'particle': PARTICLES, 'x': X_COORDS, 'y': Y_COORDS})
# print(PARTICLE_DATA)

# maximum displacement to link particles
max_displacement = 5

# link particles across frames (memory = number of frames for a particle to disappear before being excluded)
linked_particles = tp.link_df(PARTICLE_DATA, search_range=max_displacement, memory=1)

# Filter tracks based on minimum length
min_track_length = 890
filtered_tracks = tp.filter_stubs(linked_particles, min_track_length)


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# # group by particles
grouped_trajectories = filtered_tracks.groupby('particle')
FRAMES_RADII_ANGLES = []
for particle_id, trajectory in grouped_trajectories:
    X_Y = trajectory.values[:,2:] - MIDDLE
    DUMMY=[] # for storing radii and angles
    for i in range(len(X_Y)):
        # find angle
        r = np.sqrt(X_Y[i,0]**2 + X_Y[i,1]**2)
        # print()
        # print(r,X_Y[i])
        if X_Y[i,1] >= 0:
            # print(f'y>0')
            phi = np.arccos(X_Y[i,0]/r)
        else:
            # print(f'y<0')
            # print(f'arccos={np.arccos(X_Y[i,0]/r)}')
            phi = 2*np.pi - np.arccos(X_Y[i,0]/r)
        # print(phi)
        DUMMY.append((i,r,phi))
    FRAMES_RADII_ANGLES.append(DUMMY)
# exit()
# calculate dphi/dt
R_DPHI=[]
for f_r_phi in FRAMES_RADII_ANGLES:
    f_r_phi = np.array(f_r_phi)
    DUMMY=[]
    for i in range(len(f_r_phi)-30):
        dphi = f_r_phi[i+30,2] - f_r_phi[i,2] #phi difference after 1 second = 30 frames
        r = f_r_phi[i,1]
        DUMMY.append((r,dphi))
    R_DPHI.append(list(DUMMY))

# average over radius and angular velocity
R_OMEGA_MEAN=[]
for r_dphi in R_DPHI:
    r_dphi = np.array(r_dphi)
    omega_mean = np.mean(r_dphi[:,1])
    r_mean = np.mean(r_dphi[:,0])
    R_OMEGA_MEAN.append((r_mean,omega_mean))
R_OMEGA_MEAN = np.array(R_OMEGA_MEAN)


# fig, ax = plt.subplots()
# ax.scatter(R_OMEGA_MEAN[:,0],R_OMEGA_MEAN[:,1])
#
# plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# show trajectories
AXIS=np.linspace(-100,100,10)
ZEROS=np.zeros(10)

fig, ax = plt.subplots(figsize=(11,11))
ax.plot(AXIS,ZEROS,c='gray',alpha=0.7,linestyle='--')
ax.plot(ZEROS,AXIS,c='gray',alpha=0.7,linestyle='--')
for particle_id, trajectory in grouped_trajectories:
    X_Y = trajectory.values[:,2:] - MIDDLE
    ax.plot(X_Y[:,0],X_Y[:,1])

ax.set_xlabel('x [px]')
ax.set_ylabel('y [px]')
# tp.plot_traj(filtered_tracks,ax=ax)
ax.set_aspect(1)
ax.grid()
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
