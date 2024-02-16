import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from scipy.optimize import curve_fit
from tqdm import tqdm

plt.rcParams['font.size'] = '16'


output_path = 'D:/Exp_Soft_Matter/results/'
# output_path = '/local/mroot/Exp_Soft_Matter/results/'


# useful lists
FRAMES=[]
PARTICLES=[]
X_COORDS=[]
Y_COORDS=[]
TIME_RADII_ANGLES = []
HOLE_R_OMEGA=[]
TIMES_OF_TURN=[]


# useful parameters
MIDDLE = (1006.23,987.48) #x,y values of middle point
px_to_micron = 2*50/397.6
dt = 1/30


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


read_positions_from_txt(output_path+'holes.txt')
# read_positions_from_txt(output_path+'particles1.txt')


# create particle dataframe
PARTICLE_DATA = pd.DataFrame({'frame': FRAMES, 'particle': PARTICLES, 'x': X_COORDS, 'y': Y_COORDS})

# --------------DEMONSTRATION---------------------
# print(PARTICLE_DATA)
# exit()
# -------------------------------------------------



# maximum displacement to link particles
max_displacement = 5

# link particles across frames (memory = number of frames for a particle to disappear before being excluded)
linked_particles = tp.link_df(PARTICLE_DATA, search_range=max_displacement, memory=1)

# Filter tracks based on minimum length
min_track_length = 890
filtered_tracks = tp.filter_stubs(linked_particles, min_track_length)

# --------------DEMONSTRATION---------------------
# # trajectories of holes
# fig, ax = plt.subplots(figsize=(11,11))
# tp.plot_traj(filtered_tracks, ax=ax)
# ax.scatter(MIDDLE[0],MIDDLE[1],c='red',s=50,marker='x')
# ax.set_aspect(1)
# ax.grid()
# ax.set_xlabel('x [px]')
# ax.set_ylabel('y [px]')
# plt.show()
# exit()
# -------------------------------------------------




# regroup data by particles
grouped_trajectories = filtered_tracks.groupby('particle')


# transform (x,y)-values to polar coords
for particle_id, trajectory in grouped_trajectories:
    X_Y = trajectory.values[:,2:] - MIDDLE
    DUMMY=[] # for storing radii and angles
    for frame in range(len(X_Y)):
        # radius
        r = np.sqrt(X_Y[frame,0]**2 + X_Y[frame,1]**2)
        # angle
        if X_Y[frame,1] >= 0:
            phi = np.arccos(X_Y[frame,0]/r)
        else:
            phi = 2*np.pi - np.arccos(X_Y[frame,0]/r)
        DUMMY.append((frame*dt,r,phi)) #(time,radius,angle)
    TIME_RADII_ANGLES.append(DUMMY)



# change of angle
skip_num = 15 # to avoid noise
for t_r_phi in TIME_RADII_ANGLES: #for each hole
    DUMMY=[]
    t_r_phi = np.array(t_r_phi)
    r_mean = np.mean(t_r_phi[:,1]) #mean radius for comparison
    DPHI_skipped = t_r_phi[skip_num:,2] - t_r_phi[:-skip_num,2]
    DPHI = t_r_phi[1:,2] - t_r_phi[:-1,2]
    for dphi in range(len(DPHI_skipped)-1):
        if (DPHI_skipped[dphi] != 0 and (DPHI_skipped[dphi+1]/DPHI_skipped[dphi]) < 0):
            DUMMY.append(t_r_phi[dphi,0])
            print('CHANGE AT')
            print(f'{round(t_r_phi[dphi+1,0],2)}s')
            print('--------------')
    TIMES_OF_TURN.append(DUMMY)
    omega_mean = np.mean(np.abs(DPHI))/dt #mean angular velocity for comparison
    HOLE_R_OMEGA.append((r_mean,omega_mean))
HOLE_R_OMEGA = np.array(HOLE_R_OMEGA)
omega = np.mean(HOLE_R_OMEGA[:,1])

TIMES_BETWEEN_TURNS=[]

for turn_times in TIMES_OF_TURN:
    turn_times = np.array(turn_times)
    time_between_turns = turn_times[1:] - turn_times[:-1]
    TIMES_BETWEEN_TURNS.append(list(time_between_turns))

print('\n-----------------------------TIME-PASSED-BETWEEN-DIRECTION-CHANGE-------------------------------\n')
for idx,TIMES in enumerate(TIMES_BETWEEN_TURNS):
    print(f'hole Nr.{idx+1}: {TIMES}')
print()
print(f'mean angular velocity of disc: {omega} s^-1')
print('\n------------------------------------------------------------------------------------------------\n')
