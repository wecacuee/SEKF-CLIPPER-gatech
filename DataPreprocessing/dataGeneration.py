#%%
import os
import os.path as osp
import warnings
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation
import pickle
from scipy.linalg import expm,inv,block_diag


#%%
# Parameters Selection (you only need to modify this part):

np.random.seed(0)
use_normalized_camera = True
use_monocular = True

# if use_normalized_camera = False then M_file is needed
#%% 
# Data Needed
# Intrinsics Matrix M_left:
# path to KITTI intrinsics data
if not use_normalized_camera:
    M_file = osp.join((osp.dirname(__file__) or "."),
                        "..", "stereoData8", "calib.txt")

if not use_monocular:
    # base line
    baseLine = 0.54

uv_sigma_scale = 0.05 if use_normalized_camera else 5
uv_sigma = uv_sigma_scale*np.eye(2 if use_monocular else 4) # perturbation covariance matrix of u v pixel coordinates (in pixels)

MM = 400 #Number of time steps for the eight-shaped path (preferably above 40 steps)
half_path = 10 #in meters
# covariance for relative poses' positions and rotations respectively:
sigma_pose_perturbation = 0.015
sigma_rotation_perturbation = 0.01

N = 50 # Number of features

camera_limit = 10 # camera vision front limit in meters


prior_sigma = 0.7 # Covariance of the prior features positions for initialization if the triangulation is not used (in meters)

# Path where generated data is to be saved.
save_path = osp.join((osp.dirname(__file__) or "."), '..', 'stereoData8_distributed')

#%%
# Functions
def batch_invert_poses(poses):
    """Inverting Pose Matrices"""
    inverted_poses = np.zeros(poses.shape)
    inverted_rotation = np.array([np.transpose(i) for i in poses [:,0:3,0:3]])
    inverted_poses[:,0:3,0:3] = inverted_rotation
    inverted_poses[:,0:3,3] = -np.squeeze(np.matmul(inverted_rotation,np.expand_dims(poses[:,0:3,3],2)))
    inverted_poses[:,3,3] = np.ones(poses.shape[0])
    return inverted_poses


def hatMapR3(v):
    """Hat Map for R3 Vectors"""
    
    hat = np.zeros((v.shape[1],3,3))
    hat[:,0,1] = -v[2,:].T
    hat[:,0,2] = v[1,:].T
    hat[:,1,2] = -v[0,:].T
    hat[:,1,0] = -hat[:,0,1]
    hat[:,2,0] = -hat[:,0,2]
    hat[:,2,1] = -hat[:,1,2]
    return hat

def hatMapR6(v):
    """Hat Map for R6 Vectors"""

    hat = np.zeros((v.shape[1],4,4))
    hat[:,0:3,0:3] = hatMapR3(v[3::,:])
    hat[:,0:3,3] = v[0:3,:].T
    return hat

def monocular_obs_model(T,map):
    """observation model of monocular camera
    Note that matrices M is not defined internally
    """
    P = np.append(np.eye(3),np.zeros((3,1)),axis=1)
    aa = projection_pi(np.matmul(T,map))
    z = np.matmul(M, np.matmul(P, aa))
    z = np.delete(z,2,0)
    return z

def stereo_obs_model(T,map_,M):
    """observation model of monocular camera
    """
    aa = projection_pi(np.matmul(T,map_))
    z = M @ aa
    return z

def projection_pi(q):
    '''
    projection_pi function: normalizes a victor depending on its 3rd element.
    '''
    # check if z is initialized 0 and change it to one
    check_z0 = np.logical_and(q[2,:]<0.001, q[2,:]>-0.001)
    q[2,check_z0] = 0.001
    r = q/q[2,:]
    return r

if not use_normalized_camera:
    M_left = np.genfromtxt(M_file, delimiter = ' ',max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
    M_left = M_left.reshape(3,4)[:,0:3]

    # M_right
    M_right = np.genfromtxt(M_file, delimiter = ' ', skip_header=1, max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
    M_right = M_right.reshape(3,4)[:,0:3]
    img_shape = (2*M_left[0, 2], 2*M_left[1, 2]) # width(x), height(y)
else:

    M_left = np.eye(4)
    M_right = np.eye(4)
    img_shape = (np.sqrt(0.5), np.sqrt(0.5)) # width(x), height(y)

# Stereo Intrinsics
if use_monocular:
    M = np.zeros((2,4))
    M[:2,:3] = M_left[:2,:3]
else:
    M = np.zeros((4,4))
    M[:2,:3] = M_left[:2,:3]
    M[2:4,:3] = M_right[:2,:3]
    M[2,3] = -M_right[0,0]*baseLine

# %%
# Generate Ground Truth Trajectory
# x values
x1 = np.linspace(0,half_path,int(MM/4))
x2 = x1[:-1] - half_path
x = np.append(x2,x1)

# y values
r = half_path/2
y = np.zeros(len(x))
n = math.ceil(len(x)/2)
# √(r² - (x+r)²)
y[0:n] = np.array([((r**2-(x[0:n]+r)**2))**0.5])
y[n:] = -y[1:n]


# Translation Ground Truth
xy_groundTruth = np.zeros((2,2*len(x)-1))
xy_groundTruth[0,:len(x)] = x
xy_groundTruth[1,:len(x)] = y
xy_groundTruth[0,len(x):] = x[:-1][::-1]
xy_groundTruth[1,len(x):] = -y[:-1][::-1]

#%%
# Rotation Ground Truth
theta = np.zeros(xy_groundTruth.shape[1])
for i in np.arange(len(xy_groundTruth[0])):
    if xy_groundTruth[0,i] >= 0:
        a = r
        x = abs(xy_groundTruth[0,i] - a)
        y = abs(xy_groundTruth[1,i])
        angle = math.atan(y/x)
        if xy_groundTruth[0,i] > a:
            if xy_groundTruth[1,i] > 0:
                theta[i] = 2*math.pi - angle
            else:
                theta[i] = angle
        else:
            if xy_groundTruth[1,i] > 0:
                theta[i] = math.pi + angle
            else:
                theta[i] = math.pi - angle
    else:
        a = -r
        x = abs(xy_groundTruth[0,i] - a)
        y = abs(xy_groundTruth[1,i])
        angle = math.atan(y/x)
        if xy_groundTruth[0,i] > a:
            if xy_groundTruth[1,i] > 0:
                theta[i] = math.pi - angle
            else:
                theta[i] = math.pi + angle
        else:
            if xy_groundTruth[1,i] > 0:
                theta[i] = angle
            else:
                theta[i] = 2*math.pi - angle

R_groundTruth = np.zeros((xy_groundTruth.shape[1],3,3))
R_groundTruth[:,:,:] = np.eye(3)
R_groundTruth[:,0,0] = R_groundTruth[:,1,1] = np.cos(theta)
R_groundTruth[:,0,1] = np.sin(theta)
R_groundTruth[:,1,0] = -R_groundTruth[:,0,1]

#%%
# Poses:
poses_gt = np.zeros((xy_groundTruth.shape[1],4,4))
poses_gt[:,:,:] = np.eye(4)
poses_gt[:,:3,:3] = R_groundTruth
poses_gt[:,:2,3] = xy_groundTruth.T

poses_gt_2 = poses_gt.copy()
poses_gt_2[:, :2, 3] += half_path/10

poses_gt_3 = poses_gt.copy()
poses_gt_3[:, :2, 3] -= half_path/10


# Transformation from body to optical frame
opt_T_b = np.zeros((4,4))
opt_T_b[0,0] = opt_T_b[2,1] = opt_T_b[3,3] = 1
opt_T_b[1,2] = -1

def compute_perturbed_relative_poses(poses_gt):
    # Inverse Poses:
    invPoses_gt = batch_invert_poses(poses_gt)

    #%%
    # Relative Poses:
    # Relative Poses (will be used as inputs)
    inverted_poses1 = np.delete(invPoses_gt,0, axis = 0) # Tinvₜ, t ∈ (1, T)
    poses1 = np.delete(poses_gt, -1, axis = 0) # Tₜ₋₁, t ∈ (1, T)
    relative_poses = np.matmul(inverted_poses1, poses1) # Trelₜ = Tinvₜ @ Tₜ₋₁, 

    # Perturbing Relative Poses
    noise1 = np.random.normal(0,sigma_pose_perturbation,(3,relative_poses.shape[0]))
    noise2 = np.random.normal(0,sigma_rotation_perturbation,(3,relative_poses.shape[0]))
    noise = np.append(noise1, noise2, axis=0)
    #noise[3:,:] = np.zeros((noise[3:,:].shape))
    noise_hat = hatMapR6(noise)
    noise_poses =  np.array(list(map(lambda i: expm(i), noise_hat)))
    perturbed_relative_poses = np.matmul(noise_poses, relative_poses)
    return perturbed_relative_poses

perturbed_relative_poses = compute_perturbed_relative_poses(poses_gt)
perturbed_relative_poses_2 = compute_perturbed_relative_poses(poses_gt_2)
perturbed_relative_poses_3 = compute_perturbed_relative_poses(poses_gt_3)

def plot_perturbed_relative_poses(xy_groundTruth, perturbed_relative_poses):
    # check plot of Perturbed Relative Poses
    xy = np.zeros((len(xy_groundTruth[0]),3))
    T = np.eye(4)
    T[0:2,3] = -xy_groundTruth[:,0]
    xy[0,:] = inv(T)[0:3,3]

    for i in np.arange(len(perturbed_relative_poses)):
        T = np.matmul(perturbed_relative_poses[i,:,:],T)
        xy[i+1,:] = inv(T)[0:3,3]


    plt.plot(xy[:,0],xy[:,1])
    plt.scatter(xy[0,0], xy[0,1])
    plt.title('perturbed path')
    plt.show()

plot_perturbed_relative_poses(xy_groundTruth, perturbed_relative_poses)
plot_perturbed_relative_poses(poses_gt_2[:, :2, 3].T, perturbed_relative_poses_2)
plot_perturbed_relative_poses(poses_gt_3[:, :2, 3].T, perturbed_relative_poses_3)

#%%
# # Features Generation:
# x_feature_values = np.random.uniform(-half_path*1.2,half_path*1.2,N)
# y_feature_values = np.random.uniform(-half_path*0.6,half_path*0.6,N)
# z_feature_values = np.random.uniform(0,2,N)

# features = np.zeros((3,N))
# features[0,:] = x_feature_values
# features[1,:] = y_feature_values
# features[2,:] = z_feature_values


# plt.plot(xy_groundTruth[0,:],xy_groundTruth[1,:])
# plt.scatter(features[0], features[1], s=3)
# plt.title('features ground truth')
# plt.show()

#%%
n = 1
N = int(n*len(poses_gt))
N = poses_gt[::2,:3,3].shape[0]
xy_mean = np.zeros(3) # Consider changing the y mean to somewhere above 0
xy_sigma = np.diag([1,1,1])
landmarks_distrib = np.random.multivariate_normal(xy_mean,xy_sigma,N)
#features = landmarks_distrib + np.repeat(poses_gt[:,:3,3],n,axis=0) #landmarks positions in world frame
features = landmarks_distrib + poses_gt[::2,:3,3]
features = features.T

plt.plot(xy_groundTruth[0,:],xy_groundTruth[1,:])
plt.scatter(features[0], features[1], s=3)
plt.title('features ground truth')
plt.show()
#%%
# NN = int(N/6)
# # Features Generation:
# x_feature_values1 = np.random.uniform(-half_path*1.5,-half_path*1.2,NN)
# y_feature_values1 = np.random.uniform(-half_path*1,half_path*1,NN)
# z_feature_values1 = np.random.uniform(0,2,NN)

# x_feature_values2 = np.random.uniform(-half_path*1.5,half_path*1.5,int(2*NN))
# y_feature_values2 = np.random.uniform(half_path*0.7,half_path*1,int(2*NN))
# z_feature_values2 = np.random.uniform(0,2,int(2*NN))

# x_feature_values3 = np.random.uniform(half_path*1.2,half_path*1.5,NN)
# y_feature_values3 = np.random.uniform(-half_path*1,half_path*1,NN)
# z_feature_values3 = np.random.uniform(0,2,NN)

# x_feature_values4 = np.random.uniform(-half_path*1.5,half_path*1.5,int(2*NN))
# y_feature_values4 = np.random.uniform(-half_path*1,-half_path*0.7,int(2*NN))
# z_feature_values4 = np.random.uniform(0,2,int(2*NN))

# x_feature_values = np.concatenate((x_feature_values1,x_feature_values2,x_feature_values3,x_feature_values4))
# y_feature_values = np.concatenate((y_feature_values1,y_feature_values2,y_feature_values3,y_feature_values4))
# z_feature_values = np.concatenate((z_feature_values1,z_feature_values2,z_feature_values3,z_feature_values4))

# features = np.zeros((3,6*NN))
# features[0,:] = x_feature_values
# features[1,:] = y_feature_values
# features[2,:] = z_feature_values

# plt.plot(xy_groundTruth[0,:],xy_groundTruth[1,:])
# plt.scatter(features[0], features[1], s=3)
# plt.title('features ground truth')
# plt.show()

#%%
# #Mapping Features
# x_feature_values = np.random.uniform(-half_path*0.3,half_path*0.3,N)
# y_feature_values = np.random.uniform(-half_path*0.6,half_path*0.6,N)
# z_feature_values = np.random.uniform(0,2,N)

# features = np.zeros((3,N))
# features[0,:] = x_feature_values
# features[1,:] = y_feature_values
# features[2,:] = z_feature_values


# plt.plot(xy_groundTruth[0,:],xy_groundTruth[1,:])
# plt.scatter(features[0], features[1], s=3)
# plt.title('features ground truth')
# plt.show()
#%% 
# Which Features are seen at each frame

# Constraints
def generate_feature_message_positions(features, poses_gt):
    """
    Returns features_messages_positions for each timestep
    [
    { landmark_id_1 : landmark_uv_1,
    landmark_id_2 : landmark_uv_2,
    }
    ]
    """
    invPoses_gt = batch_invert_poses(poses_gt)
    img_bounds_pxls = np.array([[img_shape[0]*0.02, img_shape[1]*0.02, 1],
                                [img_shape[0]*0.98, img_shape[1]*0.98, 1]])
    img_bounds_normed = np.linalg.solve(M_left[:3, :3], img_bounds_pxls.T).T
    x_lims = img_bounds_normed[:, 0]
    y_lims = img_bounds_normed[:, 1]
    z_lims = (0, # Point should be infront of camera BINGO!
              camera_limit) # Make it a realistic road with buildings etc.
    # Note that I reduced the camera range by 50/10 pixels from each edge to account for noise 
    # that will be added to generate the features.
    homo_features = np.append(features,np.ones((1,len(features[0]))),axis=0) # landmarks in normalized coordinates
    features_messages_positions = []
    for i in np.arange(len(poses_gt)):
        rel_features = opt_T_b @ invPoses_gt[i] @ homo_features #poses in camera frame (step i)
        rel_features_normed = projection_pi(rel_features)
        x_constraint = (x_lims[0] < rel_features_normed[0, :]) & (rel_features_normed[0, :] < x_lims[1])
        y_constraint = (y_lims[0] < rel_features_normed[1, :]) & (rel_features_normed[1, :] < y_lims[1])
        z_constraint = (z_lims[0] < rel_features[2,:]) & (rel_features[2,:] < z_lims[1])
        constraint = x_constraint & z_constraint & y_constraint
        if not len(constraint):
            warnings.warn("Empty constraint")
        (ID_seen,) = np.where(constraint)
        message = {ID_seen[x]:homo_features[:,ID_seen[x]]
                   for x in np.arange(len(ID_seen))}
        features_messages_positions.append(message)
    #%%
    #Check previous step

    # Is there a feature with no appearances?
    appearing_features = [set(x.keys()) for x in features_messages_positions]
    appearing_features = list(set().union(*appearing_features)) #all features that appear

    # Is there a pose in which no feature appears?
    feats_in_pose = [len(list(x.keys())) for x in features_messages_positions]
    # assert sum(np.array(feats_in_pose) < 1) == 0 # True if all poses have features
    return features_messages_positions

features_messages_positions = generate_feature_message_positions(features, poses_gt)
features_messages_positions_2 = generate_feature_message_positions(features, poses_gt_2)
features_messages_positions_3 = generate_feature_message_positions(features, poses_gt_3)

# Check Plot at pose k:
fig, ax = plt.subplots(1,1)
artist_list = []
for k in range(len(features_messages_positions)):
    appearing_at_k = list(features_messages_positions[k].keys())
    appearing_at_k_2 = list(features_messages_positions_2[k].keys())
    appearing_at_k_3 = list(features_messages_positions_3[k].keys())
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-1.5*half_path, 1.5*half_path)
    ax.set_ylim(-0.75*half_path, 0.75*half_path)
    artists = []
    artists.extend(ax.plot(xy_groundTruth[0,:],xy_groundTruth[1,:], color='r'))
    artists.extend(ax.plot(poses_gt_2[:, 0, 3].T, poses_gt_2[:, 1, 3].T, color='g'))
    artists.extend(ax.plot(poses_gt_3[:, 0, 3].T, poses_gt_3[:, 1, 3].T, color='b'))

    artists.append(ax.scatter(features[0,appearing_at_k], features[1,appearing_at_k], s=10, color='r'))
    artists.append(ax.scatter(features[0,appearing_at_k_2], features[1,appearing_at_k_2], s=10, color='g'))
    artists.append(ax.scatter(features[0,appearing_at_k_3], features[1,appearing_at_k_3], s=10, color='b'))

    artists.append(ax.scatter(xy_groundTruth[0,k], xy_groundTruth[1,k], s = 30, color='r'))
    artists.append(ax.scatter(poses_gt_2[k, 0, 3].T, poses_gt_2[k, 1, 3].T, s = 30, color='g'))
    artists.append(ax.scatter(poses_gt_3[k, 0, 3].T, poses_gt_3[k, 1, 3].T, s = 30, color='b'))
    artists.append(ax.set_title('features seen at time k = %d' % k))
    artist_list.append(artists)
    plt.pause(0.001)

anim = animation.ArtistAnimation(fig, artist_list,interval=50)
anim.save('generatedData.mp4')

#%%
# Simulate Camera Features (u_left,v_left,u_right,v_right) with some perturbation
# Noise Question: Should the noise 2D vector be the same for all features in one frame?
# Assuming No (Independence for simplicity):
invPoses_gt = batch_invert_poses(poses_gt)
invPoses_gt_2 = batch_invert_poses(poses_gt_2)
invPoses_gt_3 = batch_invert_poses(poses_gt_3)
uv_mean = np.zeros(2 if use_monocular else 4 )
features_messages = []
features_messages_2 = []
features_messages_3 = []
for i in np.arange(len(poses_gt)):
    feats_poses = features_messages_positions[i]
    feats_poses_2 = features_messages_positions_2[i]
    feats_poses_3 = features_messages_positions_3[i]
    features_position_now = np.array(list(feats_poses.values())).T
    features_position_now_2 = np.array(list(feats_poses_2.values())).T
    features_position_now_3 = np.array(list(feats_poses_3.values())).T
    if features_position_now.size > 0:
        uv = stereo_obs_model(opt_T_b @ invPoses_gt[i],features_position_now, M)
        uv_noise = np.random.multivariate_normal(uv_mean,uv_sigma,uv.shape[1]).T
        uv_perturbed = uv + uv_noise
        uv_message = {list(feats_poses.keys())[x]:
                      uv_perturbed[:,x]
                      for x in range(len(list(feats_poses.keys())))}
        features_messages.append(uv_message)
    else:
        features_messages.append(dict())

    if features_position_now_2.size > 0:
        uv_2 = stereo_obs_model(opt_T_b @ invPoses_gt_2[i],features_position_now_2, M)
        uv_noise_2 = np.random.multivariate_normal(uv_mean,uv_sigma,uv_2.shape[1]).T
        uv_perturbed_2 = uv_2 + uv_noise_2
        uv_message_2 = {list(feats_poses_2.keys())[x]: uv_perturbed_2[:,x] for x in range(len(list(feats_poses_2.keys())))}
        features_messages_2.append(uv_message_2)
    else:
        features_messages_2.append(dict())

    if features_position_now_3.size > 0:
        uv_3 = stereo_obs_model(opt_T_b @ invPoses_gt_3[i],features_position_now_3, M)
        uv_noise_3 = np.random.multivariate_normal(uv_mean,uv_sigma,uv_3.shape[1]).T
        uv_perturbed_3 = uv_3 + uv_noise_3
        uv_message_3 = {list(feats_poses_3.keys())[x]: uv_perturbed_3[:,x] for x in range(len(list(feats_poses_3.keys())))}
        features_messages_3.append(uv_message_3)
    else:
        features_messages_3.append(dict())

# %%
# Plot of Orientation:
# Inverted Rotating Matrices
invR_groundTruth = np.array([R.T for R in R_groundTruth[:]])

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim([-half_path, half_path])
ax.set_ylim([-half_path, half_path])
ax.plot(xy_groundTruth[0,:],xy_groundTruth[1,:])

for i in np.arange(len(xy_groundTruth[0]))[::3]:
    dxx, dzx, _ = R_groundTruth[i] @ np.array([1,0,0])*0.5
    plt.arrow(xy_groundTruth[0,i], xy_groundTruth[1,i], dxx, dzx, head_width=0.1)

    dxz, dzz, _ = R_groundTruth[i] @ np.array([0,1,0])*0.5
    plt.arrow(xy_groundTruth[0,i], xy_groundTruth[1,i], dxz, dzz, head_width=0.1)
    #plt.pause(0.03)

plt.title('orientation check')
plt.show()
#%%
# Save Data:
features_msg_file = osp.join(save_path, 'features_messages')
visible_feat_file = osp.join(save_path, 'visible_features')
gt_poses_file = osp.join(save_path, 'poses')
inputs_file = osp.join(save_path, 'inputs')
features_gt_file = osp.join(save_path, 'features_gt')

# Store Features Massage:
visible_feature_ids = dict()
visible_feature_uvs = dict()
for t, fm_t in enumerate(features_messages):
    fm_t_ids = np.array(list(fm_t.keys()))
    visible_feature_ids['id_1_%d' % t] = fm_t_ids
    fm_t_uvs = (np.vstack(list(fm_t.values())) 
                if len(fm_t)
                else np.zeros((0, 2 if use_monocular else 4)))
    visible_feature_uvs['uv_1_%d' % t] = fm_t_uvs

for t, fm_t in enumerate(features_messages_2):
    fm_t_ids = np.array(list(fm_t.keys()))
    visible_feature_ids['id_2_%d' % t] = fm_t_ids
    fm_t_uvs = (np.vstack(list(fm_t.values())) 
                if len(fm_t)
                else np.zeros((0, 4)))
    visible_feature_uvs['uv_2_%d' % t] = fm_t_uvs

for t, fm_t in enumerate(features_messages_3):
    fm_t_ids = np.array(list(fm_t.keys()))
    visible_feature_ids['id_3_%d' % t] = fm_t_ids
    fm_t_uvs = (np.vstack(list(fm_t.values())) 
                if len(fm_t)
                else np.zeros((0, 4)))
    visible_feature_uvs['uv_3_%d' % t] = fm_t_uvs


outfile = open(features_msg_file,'wb')
pickle.dump([features_messages, features_messages_2, features_messages_3],outfile)
outfile.close()

np.savez(visible_feat_file,
         **visible_feature_ids,
         **visible_feature_uvs)

# Store other stuff:
np.savez(gt_poses_file, poses_gt, poses_gt_2, poses_gt_3)
np.savez(inputs_file, perturbed_relative_poses, perturbed_relative_poses_2, perturbed_relative_poses_3)
np.savez(features_gt_file, features)
# %%
