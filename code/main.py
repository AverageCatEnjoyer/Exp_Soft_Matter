import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import cv2
from tqdm import tqdm
from scipy.optimize import curve_fit


output_path = 'D:/Exp_Soft_Matter/results/'
plt.rcParams['font.size'] = '16'


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
    DUMMY_MEAN=[] # for storing radii and angles to get average value over 30 frames = 1 sec
    R_PHI_MEAN=[] # for storing radii and angles to get average value over 30 frames = 1 sec
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
        # collect averages over 30 frames
        if frame%30 == 29:
            r_phi_mean = np.zeros(2)
            r_phi_mean[0] = np.mean(np.array(R_PHI_MEAN)[:,0])
            r_phi_mean[1] = np.mean(np.array(R_PHI_MEAN)[:,1])
            DUMMY_MEAN.append((frame/30,r_phi_mean[0],r_phi_mean[1])) #time,radius,angle
            R_PHI_MEAN=[] #reset list for new values to average
    FRAMES_RADII_ANGLES.append(DUMMY)
    TIME_RADII_ANGLES_MEAN.append(DUMMY_MEAN)

TIME_RADII_ANGLES_MEAN = np.array(TIME_RADII_ANGLES_MEAN)

# # time plots of first 5 particles with averaged values
# fig, ax = plt.subplots()
# for particle,t_r_phi_mean in enumerate(TIME_RADII_ANGLES_MEAN):
#     ax.plot(t_r_phi_mean[:,0],t_r_phi_mean[:,2])
#     if particle == 100:
#         break
#
# ax.set_xlabel('t [s]')
# ax.set_ylabel('$\phi$ [rad]')
# plt.show()
# exit()

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


# exclude unrealistic data
PARTICLE_R_OMEGA_FILTERED=[]
for idx,r_omega in enumerate(PARTICLE_R_OMEGA):
    if (r_omega[0] < 400 and r_omega[1] < 0.0005):
        continue
    if (r_omega[1] < 0.008):
        PARTICLE_R_OMEGA_FILTERED.append(r_omega)
PARTICLE_R_OMEGA_FILTERED = np.array(PARTICLE_R_OMEGA_FILTERED)

# rearrange along radii
sorted_indices = np.argsort(PARTICLE_R_OMEGA_FILTERED[:, 0])
PARTICLE_R_OMEGA_FILTERED_SORTED = PARTICLE_R_OMEGA_FILTERED[sorted_indices]

# loglog fit for exponent
PARTICLE_R_OMEGA_FILTERED_SORTED_LOGLOG = np.log(PARTICLE_R_OMEGA_FILTERED_SORTED)
m,b = np.polyfit(PARTICLE_R_OMEGA_FILTERED_SORTED_LOGLOG[:,0],PARTICLE_R_OMEGA_FILTERED_SORTED_LOGLOG[:,1],deg=1)
    # result: exponent = -3.11 => -3

# function for fitting y-intercept
def omega(radius,y_intercept,r_0):
    return y_intercept + r_0*radius**-3

params = curve_fit(omega,PARTICLE_R_OMEGA_FILTERED_SORTED[:,0],PARTICLE_R_OMEGA_FILTERED_SORTED[:,1])
y_intercept = params[0][0]
r_0 = params[0][1]
print(y_intercept,r_0)
FIT_OMEGA=[]
for r_omega in PARTICLE_R_OMEGA_FILTERED_SORTED:
    FIT_OMEGA.append(omega(r_omega[0],y_intercept,r_0))
FIT_OMEGA = np.array(FIT_OMEGA)

fig, ax = plt.subplots()
# ax.scatter(PARTICLE_R_OMEGA_FILTERED[:,0],PARTICLE_R_OMEGA_FILTERED[:,1])
ax.scatter(PARTICLE_R_OMEGA_FILTERED_SORTED[:,0],PARTICLE_R_OMEGA_FILTERED_SORTED[:,1],s=15,zorder=5,label='data points')
ax.plot(np.exp(PARTICLE_R_OMEGA_FILTERED_SORTED_LOGLOG[:,0]),np.exp(m*PARTICLE_R_OMEGA_FILTERED_SORTED_LOGLOG[:,0]+b),zorder=50,label='loglog linear',linestyle='--',c='k')
ax.plot(PARTICLE_R_OMEGA_FILTERED_SORTED[:,0],FIT_OMEGA,c='red',zorder=500,label='parameters fit')
# info = f'exponent={round(m,2)}'
info = f'exponent=-3\ny-intercept={round(y_intercept,2)}\n$r_0$={round(r_0,2)}'
plt.text(300,4*10**-5,info)

ax.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('R [px]')
ax.set_ylabel('$\omega$ [$s^{-1}$]')
ax.set_title('Average angular velocity filtered results')
# ax.set_title('Average angular velocity filtered results, loglog scaling')
plt.show()


# # trajectories of ALL particles
# tp.plot_traj(filtered_tracks)
