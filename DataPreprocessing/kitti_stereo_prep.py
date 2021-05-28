#%%
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
from scipy.linalg import expm,inv,block_diag
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

def extract_block_diag(A,M):
	"""This function extracts block diagonal matrices of size M*M from the square matrix A.
	It returns a 3D array of the M*M block"""

	# Check Consitency of Sizes of A and M
	if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
		raise StandardError('Matrix must be square and a multiple of block size')

	# Extract Blocks
	blocks = np.array([A[i:i+M,i:i+M] for i in range(0,len(A)-abs(k)*M,M)])
	return blocks

def pose_inverse(poses):
	"""Inverting Pose Matrices"""
	inverted_poses = np.zeros(poses.shape)
	inverted_rotation = np.array(list(self(lambda i: np.transpose(i), poses[:,0:3,0:3])))
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
	aa = pi(np.matmul(T,map))
	z = np.matmul(M, np.matmul(P, aa))
	z = np.delete(z,2,0)
	return z

def pi(q):
	'''
	pi function: normalizes a victor depending on its 3rd element.
	'''
	# check if z is initialized 0 and change it to one
	check_z0 = np.logical_and(q[2,:]<0.001, q[2,:]>-0.001)
	q[2,check_z0] = 0.001
	r = q/q[2,:]
	return r
#%% Load Data
# kitti to world transform
wTk = np.array([[1.0,0.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,-1.0,0.0,0.0],[0.0,0.0,0.0,1.0]])
# optical to regular transform
oTc = np.array([[0,-1, 0, 0.0],[0, 0,-1, 0.0],[1, 0, 0, 0.0],[0, 0, 0, 1.0]])

# Intrinsics:
M_left = np.array([[721.5377,   0.    , 609.5593],
			 [  0.    , 721.5377, 172.854 ],
			 [  0.    ,   0.    ,   1.    ]])

# base line
baseLine = 0.54

# Stereo Intrinsics
M = np.zeros((4,4))
M[:2,:3] = M_left[:2,:3]
M[2:4,:3] = M_left[:2,:3]
M[2,3] = -M_left[0,0]*baseLine
# Load Ground Truth Trajectory
#path = "C:\\Users\\maraq\\Desktop\\Master\\EKF_project\\code\\SchmidtClear\\data_Nikolay\\car-sim\\car-sim\\"
path = '/home/malyasee/Desktop/master/EKF_clear_project/code/SchmidtClear/data_Nikolay/car-sim/car-sim/'
kTo = np.reshape(np.loadtxt(path + 'data/poses_00.txt'),(-1,3,4))
kTo = np.hstack((kTo,np.zeros(kTo.shape[:-2]+(1,4))))
kTo[...,3,3] = 1.0
poses_gt = wTk@kTo@inv(wTk)#@oTc
poses_gt = poses_gt[:1600]

# Load Camera Measurements Simulation: zs
zs_file = 'stereo_zs_features_gt.npz'
zs = np.load(path + zs_file)
zs = zs.f.arr_0
zs = zs[:1600]
#%% Generate Perturbed Relative Poses
# Relative Poses (will be used as inputs)
# Inverse Poses:
invPoses_gt = batch_invert_poses(poses_gt)
inverted_poses1 = np.delete(invPoses_gt,0, axis = 0)
poses1 = np.delete(poses_gt, -1, axis = 0)
relative_poses = np.matmul(inverted_poses1,poses1)

# Perturbing Relative Poses
sigma_pose_perturbation = 0.007
sigma_rotation_perturbation = 0.003
noise1 = np.random.normal(0,sigma_pose_perturbation,(3,relative_poses.shape[0]))
noise2 = np.random.normal(0,sigma_rotation_perturbation,(3,relative_poses.shape[0]))
noise = np.append(noise1, noise2, axis=0)
#noise[3:,:] = np.zeros((noise[3:,:].shape))
noise_hat = hatMapR6(noise)
noise_poses =  np.array(list(map(lambda i: expm(i), noise_hat)))
perturbed_relative_poses = np.matmul(noise_poses, relative_poses)

# check plot of Perturbed Relative Poses
xy = np.zeros((len(poses_gt),3))
T = np.eye(4)
T[:2,3] = -poses_gt[0,:2,3]
xy[0,:] = inv(T)[0:3,3]

for i in np.arange(len(perturbed_relative_poses)):
	T = np.matmul(perturbed_relative_poses[i,:,:],T)
	xy[i+1,:] = inv(T)[0:3,3]

#%%
plt.plot(xy[:,0],xy[:,1],label='Perturbed')
plt.plot(poses_gt[:,0,3],poses_gt[:,1,3],label='GT')
plt.scatter(xy[0,0], xy[0,1])
plt.legend()
plt.title('perturbed path')
plt.show()

#%%
# Check Which cars are seen in the duration under consideration
# Load Car Positions
path_cars_xyz = path + 'data/car_positions_00.txt'
cars_xyz = np.genfromtxt(path_cars_xyz,delimiter=' ')
#%%
# Select cars from 
x_lower_limit = np.min(poses_gt[:,0,3])-50
x_upper_limit = np.max(poses_gt[:,0,3])+50
y_lower_limit = np.min(poses_gt[:,1,3])-50
y_upper_limit = np.max(poses_gt[:,1,3])+50
check_x = np.logical_and(cars_xyz[:,0]>x_lower_limit,cars_xyz[:,0]<x_upper_limit)
check_y = np.logical_and(cars_xyz[:,2]>y_lower_limit,cars_xyz[:,2]<y_upper_limit)
check_xy = np.logical_and(check_x,check_y)
cars_xyz = cars_xyz[check_xy]

plt.scatter(cars_xyz[:,0],cars_xyz[:,2])
plt.plot(poses_gt[:,0,3],poses_gt[:,1,3],label='GT')
plt.scatter(xy[0,0], xy[0,1])
plt.legend()
plt.title('Cars')
plt.show()
#%% Extract Features from zs
# Take zs only for cars_xyz above
zs_checked = zs[:,check_xy,...]
zs_checked = zs_checked[:,::3,...]
n = 12 #features for each car
#nn = 3 #featues to keep from car
#nn = int(n/nn)
features_messages = []
N = zs_checked.shape[1]*zs_checked.shape[2] #number of features
for i in range(len(zs_checked)):
	check = np.isfinite(zs_checked[i,:,:,0])
	valid = np.array(np.where(check))
	msg = dict()
	if valid.size > 0:
		for ind in np.nditer(valid, flags=['external_loop'], order='F'):
			msg[int(ind[0]*n+ind[1])] = zs_checked[i,ind[0],ind[1]][[0,1,3,4]]
	
	features_messages.append(msg)


#%% Data Saving
save_path = '/home/malyasee/Desktop/master/EKF_clear_project/code/SchmidtClear/myStereoKitti/'
# Save Data:
features_msg_file = save_path + 'features_messages'
gt_poses_file = save_path + 'poses'
inputs_file = save_path + 'inputs'

# Store Features Massage:
outfile = open(features_msg_file,'wb')
pickle.dump(features_messages,outfile)
outfile.close()

# Store other stuff:
np.savez(gt_poses_file, poses_gt)
np.savez(inputs_file, perturbed_relative_poses)