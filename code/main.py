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
read_positions_from_txt(output_path+'particles1.txt')


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

# group by particles
grouped_trajectories = filtered_tracks.groupby('particle')
FRAMES_RADII_ANGLES = []
TIME_RADII_ANGLES_MEAN = []
for particle_id, trajectory in grouped_trajectories:
    X_Y = trajectory.values[:,2:] - MIDDLE
    DUMMY=[] # for storing radii and angles
    DUMMY_MEAN=[] # for storing radii and angles to get average value over 15 frames = 1/2 sec
    R_PHI_MEAN=[] # for storing radii and angles to get average value over 15 frames = 1/2 sec
    for frame in range(len(X_Y)):
        # print(f'frame: {frame}')
        # radius
        r = np.sqrt(X_Y[frame,0]**2 + X_Y[frame,1]**2)
        # angle
        if X_Y[frame,1] >= 0:
            phi = np.arccos(X_Y[frame,0]/r)
        else:
            phi = 2*np.pi - np.arccos(X_Y[frame,0]/r)
        DUMMY.append((frame,r,phi))
        R_PHI_MEAN.append((r,phi))
        # collect averages over 15 frames
        if frame%15 == 14:

            # print()
            # print(f'NOW frame%15 = {frame%15}')
            # print("R_PHI_MEAN: \n",R_PHI_MEAN)
            r_phi_mean = np.zeros(2)
            r_phi_mean[0] = np.mean(np.array(R_PHI_MEAN)[:,0])
            r_phi_mean[1] = np.mean(np.array(R_PHI_MEAN)[:,1])
            # print(f"r_phi_mean={r_phi_mean}")
            DUMMY_MEAN.append((frame/30,r_phi_mean[0],r_phi_mean[1]))
    FRAMES_RADII_ANGLES.append(DUMMY)
    TIME_RADII_ANGLES_MEAN.append(DUMMY_MEAN)

TIME_RADII_ANGLES_MEAN = np.array(TIME_RADII_ANGLES_MEAN)

# calculate change of angle
# R_DPHI=[]
# for f_r_phi in FRAMES_RADII_ANGLES:
#     f_r_phi = np.array(f_r_phi)
#     DUMMY=[]
#     for i in range(len(f_r_phi)-30):
#         dphi = f_r_phi[i+30,2] - f_r_phi[i,2] #phi difference after 1 second = 30 frames
#         r = f_r_phi[i,1]
#         DUMMY.append((r,dphi))
#     R_DPHI.append(list(DUMMY))
#
# # average over radius and angular velocity
# R_OMEGA_MEAN=[]
# for r_dphi in R_DPHI:
#     r_dphi = np.array(r_dphi)
#     omega_mean = np.mean(r_dphi[:,1])
#     r_mean = np.mean(r_dphi[:,0])
#     R_OMEGA_MEAN.append((r_mean,omega_mean))
# R_OMEGA_MEAN = np.array(R_OMEGA_MEAN)
#

# mean radius and mean frequency of each particle
PARTICLE_R_OMEGA=[]
for t_r_phi_mean in TIME_RADII_ANGLES_MEAN:
    r_mean_mean = np.mean(t_r_phi_mean[:,1])
    for i in range(len(t_r_phi_mean)-1):
        DPHI = np.zeros(len(t_r_phi_mean)-1)
        dphi = t_r_phi_mean[i+1,2] - t_r_phi_mean[i,2] #phi difference after 1/2 seconds
        DPHI[i] = dphi
    omega = np.mean(np.abs(DPHI))
    PARTICLE_R_OMEGA.append((r_mean_mean,omega))
PARTICLE_R_OMEGA = np.array(PARTICLE_R_OMEGA)

fig, ax = plt.subplots()
ax.scatter(PARTICLE_R_OMEGA[:,0],PARTICLE_R_OMEGA[:,1])

ax.set_xlabel('R [px]')
ax.set_ylabel('$\omega$ [2/s]')
plt.show()

exit()

# plot of first 5 particles with averaged values
fig, ax = plt.subplots()
for particle,t_r_phi_mean in enumerate(TIME_RADII_ANGLES_MEAN):
    ax.plot(t_r_phi_mean[:,0],t_r_phi_mean[:,2])
    if particle == 5:
        break
plt.show()

# tp.plot_traj(filtered_tracks)

# convert into numpy array
# TRAJECTORIES = filtered_tracks.values
#
#
# fig, ax = plt.subplots()
# # tp.plot_traj(filtered_tracks,ax=ax)
#
# plt.show()
