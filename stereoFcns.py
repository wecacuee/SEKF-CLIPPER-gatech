import numpy as np

from utils import hatMapR6, block_diag, dpi, pi, cir, expm

#%% 
# Models

# Motion Model

def prediction_step(T_past, inputt, input_adjoint, sigma, W):
	"""Prediction step of pose and covariance"""
	T_present = inputt @ T_past
	sigma_new = input_adjoint @ sigma @ input_adjoint.T + W
	return T_present, sigma_new

# Monocular Camera Observation Model
def monocular_obs_model(T, map, P):
	"""observation model of monocular camera
	Note that matrices M and P won't be defined internally
	"""
	z = np.matmul(P, pi(np.matmul(T, map)))
	z = np.delete(z, 2, 0)
	return z

def stereo_obs_model(T,map,baseLine):
	"""observation model of monocular camera
	"""
	aa = pi(np.matmul(T,map))
	MM = np.zeros((4,4))
	MM[0,0] = MM[1,1] = MM[2,0] = MM[3,1] = 1
	MM[2,3] = -baseLine
	z = np.matmul(MM, aa)
	return z


#%%
# Jacobians of Above Models (Analytic)

def place_Jacobian(Jacobians, h, j, seen_sum):
	"""This function is used in Landmark_Jacobian()"""
	n = h[j]
	H_map = np.zeros((2, 3*seen_sum))
	H_map[:, int(n*3):int(n*3+3)] = Jacobians[j]
	return H_map

def stereo_place_Jacobian(Jacobians, h, j, seen_sum):
	"""This function is used in Landmark_Jacobian()"""
	n = h[j]
	H_map = np.zeros((4, 3*seen_sum))
	H_map[:, int(n*3):int(n*3+3)] = Jacobians[j]
	return H_map

def landmark_Jacobian(T, map_now, index_map_now, P, map_size):
	"""Generate the full jacobian of the features"""
	# Jacobian for each observation:
	Jacobians = np.matmul(P, np.matmul(
		dpi(np.matmul(T, map_now)), np.matmul(T, P.T)))
	Jacobians = np.delete(Jacobians, 2, axis=1)
	# Stack all Jacobians into one large matrix showing the correspondance
	# between observations and landmarks:
	h = index_map_now
	H_map = np.array([place_Jacobian(Jacobians, h, j, map_size)
					 for j in np.arange(Jacobians.shape[0])])
	H_map = np.reshape(H_map, (H_map.shape[0]*H_map.shape[1], H_map.shape[2]))
	return H_map

def stereo_landmark_Jacobian(T, map_now, index_map_now, P, map_size, baseLine):
	"""Generate the full jacobian of the features"""
	MM = np.zeros((4,4))
	MM[0,0] = MM[1,1] = MM[2,0] = MM[3,1] = 1
	MM[2,3] = -baseLine
	# Jacobian for each observation:
	Jacobians = np.matmul(MM, np.matmul(
		dpi(np.matmul(T, map_now)), np.matmul(T, P.T)))
	# Stack all Jacobians into one large matrix showing the correspondance
	# between observations and landmarks:
	h = index_map_now
	H_map = np.array([stereo_place_Jacobian(Jacobians, h, j, map_size)
					 for j in np.arange(Jacobians.shape[0])])
	H_map = np.reshape(H_map, (H_map.shape[0]*H_map.shape[1], H_map.shape[2]))
	return H_map

def pose_Jacobian(T, map_now, P, opt_T_body):
	"""Generate the pose Jacobian"""
	H1 = np.matmul(opt_T_body, cir(np.matmul(T, map_now)))
	# H2 = np.matmul(np.matmul(M,P),np.matmul(dpi(np.matmul(T,map_now)),H1))
	H2 = np.matmul(P, dpi(np.matmul(opt_T_body @ T, map_now)))
	H2 = np.matmul(H2, H1)
	H2 = np.delete(H2, 2, axis=1)
	H_pose = H2.reshape((int(2*len(H2)), 6))
	return H_pose

def stereo_pose_Jacobian(T, map_now, opt_T_body, baseLine):
	"""Generate the pose Jacobian"""
	MM = np.zeros((4,4))
	MM[0,0] = MM[1,1] = MM[2,0] = MM[3,1] = 1
	MM[2,3] = -baseLine
	H1 = np.matmul(opt_T_body, cir(np.matmul(T, map_now)))
	H2 = np.matmul(MM, dpi(np.matmul(opt_T_body @ T, map_now)))
	H2 = np.matmul(H2, H1)
	H_pose = H2.reshape((int(4*len(H2)), 6))
	return H_pose

#%%
# Update Step (Schmidt Extended Kalman Filter):

def schmidt_filter(H, sigma, z_hat, feats, V, updated_states_number):
	"""Gives the Kalman Gain and the innovations of pose and SLAM map"""
	n = int(H.shape[0]/V.shape[0])  # number of features
	V_diag = block_diag(*([V] * n))  # noise covarience matrix
	aa = np.matmul(H, np.matmul(sigma, H.T))
	#bb = inv(aa + V_diag)
	bb = np.linalg.solve(aa+V_diag,np.eye(V_diag.shape[0]))
	Kalman_Gain = np.zeros((sigma.shape[0], H.shape[0]))
	Kalman_Gain[:updated_states_number,
		:] = sigma[:updated_states_number, :] @ H.T @ bb
	innovation = (feats.T - z_hat).T.reshape((z_hat.size, 1))
	update_pose = Kalman_Gain[:6, :] @ innovation
	update_slam_map = Kalman_Gain[6:updated_states_number, :] @ innovation
	return Kalman_Gain, update_pose, update_slam_map

def schmidt_update(T, update_pose, slam_map, update_slam_map, sigma, Kalman_Gain, H, updated_states_number):
	"""Performs the the update step for the pose, map, and covariance sigma"""
	n = updated_states_number
	# Update Pose
	T_updated = np.matmul(expm(hatMapR6(update_pose).reshape((4, 4))), T)
	# Update slam_map
	# Reshaping to fit the update equation dimentions
	map_reshaped = np.zeros((slam_map[0:3, :].size, 1))
	map_reshaped[:,:] = slam_map[0:3, :].T.reshape((slam_map[0:3, :].size, 1))
	slam_map_updated = map_reshaped + update_slam_map
	slam_map_updated = slam_map_updated.reshape(
		(int(slam_map_updated.size/3), 3)).T  # Back to original shape
	slam_map_updated = np.append(slam_map_updated, np.ones(
		(1, slam_map_updated.shape[1])), 0)  # make homogenuous coordinates
	# Update Covariance
	#sigma_new = np.eye(sigma.shape[0])
	sigma_new = np.eye(len(sigma))
	sigma_new[:,:] = sigma[:,:]
	sigma_new[0:n, 0:n] = sigma[0:n, 0:n] - Kalman_Gain[:n,
		:] @ (H[:, :n] @ sigma[0:n, 0:n] + H[:, n:] @ sigma[n:, :n])
	sigma_new[0:n, n:] = sigma[0:n, n:] - Kalman_Gain[:n,
		:] @ (H[:, :n] @ sigma[0:n, n:] + H[:, n:] @ sigma[n:, n:])
	sigma_new[n:, 0:n] = sigma_new[0:n, n:].T
	return T_updated, slam_map_updated, sigma_new
