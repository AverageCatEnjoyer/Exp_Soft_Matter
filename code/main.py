import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from scipy.optimize import curve_fit
from tqdm import tqdm


output_path = 'D:/Exp_Soft_Matter/results/'
# output_path = '/local/mroot/Exp_Soft_Matter/results/'
plt.rcParams['font.size'] = '16'


# useful lists
FRAMES=[]
PARTICLES=[]
X_COORDS=[]
Y_COORDS=[]
TIME_RADII_ANGLES = []
PARTICLE_R_OMEGA=[]
PARTICLE_R_OMEGA_FILTERED=[]
FIT_OMEGA=[]



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


# read_positions_from_txt(output_path+'holes.txt')
read_positions_from_txt(output_path+'particles1.txt')


# create particle dataframe
PARTICLE_DATA = pd.DataFrame({'frame': FRAMES, 'particle': PARTICLES, 'x': X_COORDS, 'y': Y_COORDS})

# --------------DEMONSTRATION---------------------
# print(PARTICLE_DATA)
# exit()
# -------------------------------------------------



# maximum displacement to link particles
max_displacement = 20

# link particles across frames (memory = number of frames for a particle to disappear before being excluded)
linked_particles = tp.link_df(PARTICLE_DATA, search_range=max_displacement, memory=1)

# Filter tracks based on minimum length
min_track_length = 890
filtered_tracks = tp.filter_stubs(linked_particles, min_track_length)

# --------------DEMONSTRATION---------------------
# # trajectories of ALL particles
# fig, ax = plt.subplots(figsize=(11,11))
# tp.plot_traj(filtered_tracks, ax=ax)
# ax.scatter(MIDDLE[0],MIDDLE[1],c='red',s=50,marker='x')
# ax.set_aspect(1)
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


# mean radius and mean frequency of each particle
for t_r_phi in TIME_RADII_ANGLES:
    t_r_phi = np.array(t_r_phi)
    r_mean = np.mean(t_r_phi[:,1]) #mean radius
    DPHI = t_r_phi[1:,2] - t_r_phi[:-1,2]
    omega_mean = np.mean(np.abs(DPHI))/dt #mean angular velocity
    PARTICLE_R_OMEGA.append((r_mean,omega_mean))
PARTICLE_R_OMEGA = np.array(PARTICLE_R_OMEGA)


# exclude unrealistic data
for idx,r_omega in enumerate(PARTICLE_R_OMEGA):
    if (r_omega[1] < 0.7):
        PARTICLE_R_OMEGA_FILTERED.append(r_omega)
PARTICLE_R_OMEGA_FILTERED = np.array(PARTICLE_R_OMEGA_FILTERED)
# PARTICLE_R_OMEGA_FILTERED = PARTICLE_R_OMEGA


# rearrange along radii
sorted_indices = np.argsort(PARTICLE_R_OMEGA_FILTERED[:, 0])
PARTICLE_R_OMEGA_FILTERED_SORTED = PARTICLE_R_OMEGA_FILTERED[sorted_indices]


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


# final data
RADIUS = px_to_micron*PARTICLE_R_OMEGA_FILTERED_SORTED[:,0]
ANG_VELOCITY = PARTICLE_R_OMEGA_FILTERED_SORTED[:,1]


# loglog fit for exponent
threshold = (RADIUS>70) & (RADIUS<150)
RADIUS_LOG = np.log(RADIUS[threshold])
ANG_VELOCITY_LOG = np.log(ANG_VELOCITY[threshold])
m,b = np.polyfit(RADIUS_LOG,ANG_VELOCITY_LOG,deg=1)

print('\n\n\n-----------------------------LOGLOG-FIT------------------------------\n')
print(f'exponent={m}')
print('\n---------------------------------------------------------------------\n')


# parameter fit with extracted exponent
def omega(radius,y_intercept,r_0):
    radius = radius/r_0
    return y_intercept + radius**-3

params = curve_fit(omega,RADIUS[threshold],ANG_VELOCITY[threshold])
y_intercept = params[0][0]
r_0 = params[0][1]

for radius in RADIUS:
    FIT_OMEGA.append(omega(radius,0,r_0))
FIT_OMEGA = np.array(FIT_OMEGA)

print('\n-----------------------------PARAMETER-FIT-----------------------------\n')
print(f'y-intercept={y_intercept}, scaling_parameter={r_0}')
print('\n-----------------------------------------------------------------------\n')



# plotting
fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(RADIUS,ANG_VELOCITY,s=15,zorder=5,label='data points')
# ax.plot(np.exp(RADIUS_LOG),np.exp(m*RADIUS_LOG+b),zorder=50,label='loglog linear',linestyle='--',c='k')
ax.plot(RADIUS,FIT_OMEGA,c='red',zorder=500,label='parameters fit')

info = f'exponent=-3\n$r_0$={round(r_0,2)}'
ax.set_title('average angular velocity results')
ax.set_xlabel('R [$\mu m$]')
ax.set_ylabel('$\omega$ [$s^{-1}$]')
ax.legend()
ax.grid()
plt.text(200,0.58,info)

# ---------log---------
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_title('average angular velocity results, loglog scaling')
# info = f'exponent={round(m,2)}'
# # info = f'exponent={round(m,2)}\n$r_0$={round(r_0,2)}'
# plt.text(150,0.5,info)


plt.show()
