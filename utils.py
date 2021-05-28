# For EKF
import numpy as np
import pickle
import os
from scipy.linalg import expm, inv, block_diag
from matplotlib import pyplot as plt
import time

# Load Data:
def load_data(path):
	# Files
	feats_messages_file = path+'features_messages'
	inputs_file = path+'inputs.npz'
	M_file = path+'calib.txt'
	prior_map_file = path+'prior_features.npz'
	features_gt_file = path+'features_gt.npz'
	poses_gt_file = path+'poses.npz'

	# Features Messages
	infile = open(feats_messages_file,'rb')
	features_messages = pickle.load(infile)
	infile.close()

	# Inputs
	inputs = np.load(inputs_file)
	inputs = inputs.f.arr_0

	# Priors
	prior_map = np.load(prior_map_file)
	prior_map = prior_map.f.arr_0

	# Features_gt
	features_gt = np.load(features_gt_file)
	features_gt = features_gt.f.arr_0

	# Poses_gt
	poses_gt = np.load(poses_gt_file)
	poses_gt = poses_gt.f.arr_0

	# M (intrinsics)
	M = np.genfromtxt(M_file, delimiter=' ', max_rows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
	M = M.reshape(3, 4)[:, 0:3]

	return features_messages, inputs, M, prior_map, features_gt, poses_gt

def load_data_kitti(path):
	# Files
	feats_messages_file = path+'features_messages'
	inputs_file = path+'inputs.npz'
	poses_gt_file = path+'poses.npz'

	# Features Messages
	infile = open(feats_messages_file,'rb')
	features_messages = pickle.load(infile)
	infile.close()

	# Inputs
	inputs = np.load(inputs_file)
	inputs = inputs.f.arr_0

	# Poses_gt
	poses_gt = np.load(poses_gt_file)
	poses_gt = poses_gt.f.arr_0


	M = np.array([[721.5377,   0.    , 609.5593],
				[  0.    , 721.5377, 172.854 ],
				[  0.    ,   0.    ,   1.    ]])    
	
	return features_messages, inputs, M, poses_gt

def stereo_load_data(path):
	# Files
	feats_messages_file = path+'features_messages'
	inputs_file = path+'inputs.npz'
	M_file = path+'calib.txt'
	features_gt_file = path+'features_gt.npz'
	poses_gt_file = path+'poses.npz'

	# Features Messages
	infile = open(feats_messages_file,'rb')
	features_messages = pickle.load(infile)
	infile.close()

	# Inputs
	inputs = np.load(inputs_file)
	inputs = inputs.f.arr_0

	# Features_gt
	features_gt = np.load(features_gt_file)
	features_gt = features_gt.f.arr_0

	# Poses_gt
	poses_gt = np.load(poses_gt_file)
	poses_gt = poses_gt.f.arr_0

	# M (intrinsics)
	# Intrinsics Matrix M_left:
	M_left = np.genfromtxt(M_file, delimiter = ' ',max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
	M_left = M_left.reshape(3,4)[:,0:3]

	# M_right
	M_right = np.genfromtxt(M_file, delimiter = ' ', skip_header=1, max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
	M_right = M_right.reshape(3,4)[:,0:3]

	# base line
	baseLine = 0.54

	# Stereo Intrinsics
	M = np.zeros((4,4))
	M[:2,:3] = M_left[:2,:3]
	M[2:4,:3] = M_right[:2,:3]
	M[2,3] = -M_right[0,0]*baseLine

	return features_messages, inputs, M, features_gt, poses_gt, baseLine

def stereo_load_data_kitti(path):
	# Files
	feats_messages_file = path+'features_messages'
	inputs_file = path+'inputs.npz'
	poses_gt_file = path+'poses.npz'

	# Features Messages
	infile = open(feats_messages_file,'rb')
	features_messages = pickle.load(infile)
	infile.close()

	# Inputs
	inputs = np.load(inputs_file)
	inputs = inputs.f.arr_0

	# Poses_gt
	poses_gt = np.load(poses_gt_file)
	poses_gt = poses_gt.f.arr_0


	M_left = M_right = np.array([[721.5377,   0.    , 609.5593],
				[  0.    , 721.5377, 172.854 ],
				[  0.    ,   0.    ,   1.    ]])    
	
	baseLine = 0.54

	# Stereo Intrinsics
	M = np.zeros((4,4))
	M[:2,:3] = M_left[:2,:3]
	M[2:4,:3] = M_right[:2,:3]
	M[2,3] = -M_right[0,0]*baseLine
	
	return features_messages, inputs, M, poses_gt, baseLine

def initialize_parameters(initial_pose, inputs, map_prior):
	"""
	Initialize parameters for filtering
	"""
	time_steps = len(inputs) + 1

	# Pose Estimates Storage
	T = np.zeros((time_steps, 4, 4))  # poses state
	T[0, :, :] = initial_pose

	# Inputs Adjoints
	inputs_adjoints = pose_adjoint(inputs)
	
	# prior map in homogeneous coordinates
	map_prior_new = np.append(
		map_prior, np.ones((1, map_prior.shape[1])), 0)

	return T, inputs_adjoints, map_prior_new

def stereo_initialize_parameters(initial_pose, inputs):
	"""
	Initialize parameters for filtering
	"""
	time_steps = len(inputs) + 1

	# Pose Estimates Storage
	T = np.zeros((time_steps, 4, 4))  # poses state
	T[0, :, :] = initial_pose

	# Inputs Adjoints
	inputs_adjoints = pose_adjoint(inputs)
	
	return T, inputs_adjoints

def initialize_parameters_kitti(initial_pose, inputs):
	"""
	Initialize parameters for filtering
	"""
	time_steps = len(inputs) + 1

	# Pose Estimates Storage
	T = np.zeros((time_steps, 4, 4))  # poses state
	T[0, :, :] = initial_pose

	# Inputs Adjoints
	inputs_adjoints = pose_adjoint(inputs)
	
	return T, inputs_adjoints


#%%

def pix2normalized(features_msg_dict, M_inv):
	for k in features_msg_dict.keys():
		norm_uv = M_inv @ np.array([*features_msg_dict[k],1])
		features_msg_dict[k] = norm_uv[:2]
	return features_msg_dict

def stereo_pix2normalized(features_msg_dict, M):
	for k in features_msg_dict.keys():
		norm_uv = (features_msg_dict[k] - M[:,2])/M[[0,1,2,3],[0,1,0,1]]
		features_msg_dict[k] = norm_uv
	return features_msg_dict

def invert_pose(pose):
	"""Invert one pose (not batch)"""
	inverted_pose = np.eye(4)
	inverted_pose[:3, :3] = pose[:3, :3].T
	inverted_pose[:3, 3] = -inverted_pose[:3, :3] @ pose[:3, 3]
	return inverted_pose

def pose_inverse(poses):
	"""Inverting a batch of N Pose Matrices
	poses: N*4*4 array containing N pose matrices"""
	inverted_poses = np.zeros(poses.shape)
	inverted_rotation = np.array(
		list(map(lambda i: np.transpose(i), poses[:, 0:3, 0:3])))
	inverted_poses[:, 0:3, 0:3] = inverted_rotation
	inverted_poses[:, 0:3, 3] = - \
		np.squeeze(np.matmul(inverted_rotation,
				   np.expand_dims(poses[:, 0:3, 3], 2)))
	inverted_poses[:, 3, 3] = np.ones(poses.shape[0])
	return inverted_poses


def hatMapR3(v):
	"""Hat Map for a batch of R3 Vectors"""

	hat = np.zeros((v.shape[1], 3, 3))
	hat[:, 0, 1] = -v[2, :].T
	hat[:, 0, 2] = v[1, :].T
	hat[:, 1, 2] = -v[0, :].T
	hat[:, 1, 0] = -hat[:, 0, 1]
	hat[:, 2, 0] = -hat[:, 0, 2]
	hat[:, 2, 1] = -hat[:, 1, 2]
	return hat


def hatMapR6(v):
	"""Hat Map for a batch of R6 Vectors"""

	hat = np.zeros((v.shape[1], 4, 4))
	hat[:, 0:3, 0:3] = hatMapR3(v[3::, :])
	hat[:, 0:3, 3] = v[0:3, :].T
	return hat


def pose_adjoint(inputs):
	"""Givest the adjoint map of a batch of poses matrices here called 'inputs'"""
	adj = np.zeros((inputs.shape[0], 6, 6))
	adj[:, 0:3, 0:3] = adj[:, 3:6, 3:6] = inputs[:, 0:3, 0:3]
	adj[:, 0:3, 3:6] = np.matmul(hatMapR3(
		inputs[:, 0:3, 3].T.reshape((3, inputs.shape[0]))), inputs[:, 0:3, 0:3])
	return adj


def cir(m):
	# This Function does what the function of circle and a dot inside it does in the
	# pose Jacobian
	cir = np.zeros((m.shape[1], 4, 6))
	cir[:, 0:3, 0:3] = np.identity(3)
	cir[:, 0:3, 3:6] = -hatMapR3(m[0:3, :])
	return cir


def pi(q):
	'''
	pi function: normalizes a victor depending on its 3rd element.
	'''
	# check if z is initialized 0 and change it to one
	check_z0 = np.logical_and(q[2, :] < 0.00001, q[2, :] > -0.00001)
	q[2, check_z0] = 0.00001
	r = q/q[2, :]
	return r


def dpi(q):
	# Differentiation of the pi function
	r = -pi(q)
	mat = np.zeros((q.shape[1], 4, 4))
	mat[:] = np.identity(4)
	mat[:, :, 2] = r.T
	mat[:, 2, 2] = 0
	mat = [mat[i]/(q[2, :][i]) for i in range(len(q[2, :]))]
	return np.array(mat)


def numericalJacobian(funhandle, addhandle, x, J):
	'''
	funhandle(x) = n x 1
	J = n x d = Jacobian of funhandle at x
	'''
	delta = 0.000001
	d = J.shape[-1]
	dx = delta*np.eye(d)
	Jhat = np.zeros(J.shape)
	for k in range(d):
		Jhat[...,k] = ((funhandle(addhandle(dx[k],x))-funhandle(x))/delta).squeeze()
	return Jhat


def R6plusSE3(zeta,T):
	r"""
	left multiplication of exp(\hat{\zeta}) by T
	"""
	zhat = hatMapR6(zeta[...,None]).squeeze()
	Tz = expm(zhat)
	new_T = Tz @ T
	return new_T


def triangulate(T11,T22,obs1,obs2,opt_T_body):
	"""Triangulation function used for initialization by triangulation
	see the Feature object in mapClass.py"""
	T1 = invert_pose(opt_T_body @ T11)
	T2 = invert_pose(opt_T_body @ T22)
	T12 = invert_pose(T1) @ T2

	v1 = (np.append(obs1, np.array([1]),axis=0)).reshape((3,1))
	v1 = v1/np.linalg.norm(v1)
	v2 = (np.append(obs2, np.array([1]),axis=0)).reshape((3,1))
	v2 = v2/np.linalg.norm(v2)

	A = np.append(v1, -T12[:3,:3] @ v2, axis=1) #check
	b = T12[:3,3]

	depth = np.linalg.lstsq(A,b,rcond = None)
	p1 = depth[0][0] * v1
	pf_1 = T1 @ np.append(p1, np.array([[1]]),axis=0)

	p2 = depth[0][1] * v2
	pf_2 = T2 @ np.append(p2, np.array([[1]]),axis=0)

	pf = 0.5*(pf_1 + pf_2)[:-1]

	return pf


def stereo_triangulate(T,obs,opt_T_body,baseLine):
	"""Triangulation function used for initialization by triangulation
	see the Feature object in mapClass.py"""
	T = invert_pose(opt_T_body @ T)
	pf_opt = np.zeros((4,1))
	#depth
	d = baseLine/(obs[0]-obs[2]) #depth
	#Relative position (optical frame)
	pf_opt[0,0] = obs[0]*d
	pf_opt[1,0] = (obs[1]+obs[3])/2*d
	pf_opt[2,0] = d
	pf_opt[3,0] = 1
	# position in world frame
	pf = T @ pf_opt
	pf = pf[:3]
	return pf