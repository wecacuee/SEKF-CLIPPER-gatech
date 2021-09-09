#%%
import os
import os.path as osp
import warnings
import math
import json
import pickle

from argparse import Namespace

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.linalg import expm,inv,block_diag


def generate_config():
    C = Namespace
    #%%
    # Parameters Selection (you only need to modify this part):

    C.seed = 0
    C.use_normalized_camera = True
    C.use_monocular = True
    C.num_robots = 6
    C.colors = 'rbgcbyk'

    # if use_normalized_camera = False then M_file is needed
    #%% 
    # Data Needed
    # Intrinsics Matrix M_left:
    # path to KITTI intrinsics data
    if not C.use_normalized_camera:
        C.M_file = osp.join((osp.dirname(__file__) or "."),
                            "..", "stereoData8", "calib.txt")

    if not C.use_monocular:
        # base line
        C.baseLine = 0.54

    C.uv_sigma_scale = 0.05 if C.use_normalized_camera else 5
    C.uv_sigma = C.uv_sigma_scale*np.eye(2 if C.use_monocular else 4) # perturbation covariance matrix of u v pixel coordinates (in pixels)

    C.MM = 400 #Number of time steps for the eight-shaped path (preferably above 40 steps)
    C.half_path = 10 #in meters
    # covariance for relative poses' positions and rotations respectively:
    C.sigma_pose_perturbation = 0.015
    C.sigma_rotation_perturbation = 0.01

    C.N = 50 # Number of features

    C.camera_limit = 10 # camera vision front limit in meters


    C.prior_sigma = 0.7 # Covariance of the prior features positions for initialization if the triangulation is not used (in meters)

    # Path where generated data is to be saved.
    C.save_path = osp.join((osp.dirname(__file__) or "."), '..', 'multi-robot-data-%d' % C.num_robots)
    if not osp.exists(C.save_path):
        os.makedirs(C.save_path)

    # Transformation from body to optical frame
    C.opt_T_b = np.zeros((4,4))
    C.opt_T_b[0,0] = C.opt_T_b[2,1] = C.opt_T_b[3,3] = 1
    C.opt_T_b[1,2] = -1


    if not C.use_normalized_camera:
        C.M_left = np.genfromtxt(C.M_file, delimiter = ' ',max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
        C.M_left = C.M_left.reshape(3,4)[:,0:3]

        # M_right
        C.M_right = np.genfromtxt(C.M_file, delimiter = ' ', skip_header=1, max_rows = 1, usecols= (1,2,3,4,5,6,7,8,9,10,11,12))
        C.M_right = C.M_right.reshape(3,4)[:,0:3]
        C.img_shape = (2*C.M_left[0, 2], 2*C.M_left[1, 2]) # width(x), height(y)
    else:
        C.M_left = np.eye(4)
        C.M_right = np.eye(4)
        C.img_shape = (np.sqrt(0.5), np.sqrt(0.5)) # width(x), height(y)

    # Stereo Intrinsics
    if C.use_monocular:
        C.M = np.zeros((2,4))
        C.M[:2,:3] = C.M_left[:2,:3]
    else:
        C.M = np.zeros((4,4))
        C.M[:2,:3] = M_left[:2,:3]
        C.M[2:4,:3] = M_right[:2,:3]
        C.M[2,3] = -M_right[0,0]*baseLine

    return C

CONFIG = generate_config()

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


def batch_matmul(A, B):
    assert A.shape[-1] == B.shape[-2]
    return (A[..., :, :, None] * B[..., None, :, :]).sum(axis=-2)

class Lissajous:
    """
    Plots Lissajous curve

    x = A sin(at + δ)
    y = B sin(bt)

    Optional arguments
    """
    def __init__(s,
                 A=CONFIG.half_path,
                 B=CONFIG.half_path,
                 a=1,
                 b=2,
                 δ=np.pi/2):
        s.A = A
        s.B = B
        s.a = a
        s.b = b
        s.δ = δ

    def generate_ground_truth_trajectory(s, robot_num):
        A = s.A
        B = s.B
        a = s.a
        b = s.b
        δ = s.δ

        t = np.linspace(-np.pi, np.pi, CONFIG.MM)
        x = A * np.sin(a*t + δ)
        y = B * np.sin(b*t)
        z = np.zeros_like(t)

        θ = np.arctan2(y[1:]-y[:-1], x[1:]-x[:-1])
        θ = np.hstack([θ, np.arctan2(y[0]-y[-1], x[0] - x[-1])]).reshape(-1, 1, 1)
        n = θ.shape[0]
        θ = θ - np.pi/2 # Hack to make robots align with the trajectory
        zs = np.zeros((n, 1, 1))
        os = np.ones((n, 1, 1))
        Rθ = np.concatenate([
            np.concatenate([np.cos(θ), -np.sin(θ), zs], axis=-1),
            np.concatenate([np.sin(θ),  np.cos(θ), zs], axis=-1),
            np.concatenate([       zs,         zs, os], axis=-1),
            ], axis=-2)
        assert is_valid_so3(Rθ)
        poses_gt = np.zeros((t.shape[0], 4, 4))
        poses_gt[:, :3, :3] = Rθ
        poses_gt[:, 0, 3] = x
        poses_gt[:, 1, 3] = y
        poses_gt[:, 2, 3] = z
        poses_gt[:, 3, 3] = 1

        robot_theta = 2*np.pi*robot_num/CONFIG.num_robots
        robot_rot = np.array([
            [np.cos(robot_theta), - np.sin(robot_theta), 0],
            [np.sin(robot_theta),   np.cos(robot_theta), 0],
            [                  0,                     0, 1]])
        robot_shift = robot_rot @ np.array([A*0.5, 0, 0])
        robot_pos = np.vstack([np.hstack([robot_rot,  robot_shift.reshape(-1, 1)]),
                               np.array([[0, 0, 0, 1]])])
        assert is_valid_se3(robot_pos)
        assert is_valid_se3(poses_gt)
        poses_gt_robot = batch_matmul(robot_pos, poses_gt)
        assert is_valid_se3(poses_gt_robot)
        return poses_gt_robot

    def generate_ground_truth_trajectories(self):
        #%%
        # Poses:
        poses_gt = list()
        for r in range(CONFIG.num_robots):
            pose_gt_per_robot = self.generate_ground_truth_trajectory(r)
            poses_gt.append(pose_gt_per_robot[None, ...])
        return np.concatenate(poses_gt, axis=0)


class EightPattern:
    def __init__(s):
        pass
    def generate_ground_truth_trajectory(s, robot_num):
        # %%
        # Generate Ground Truth Trajectory
        # x values
        x1 = np.linspace(0,CONFIG.half_path,int(CONFIG.MM/4))
        x2 = x1[:-1] - CONFIG.half_path
        x = np.append(x2,x1)

        # y values
        r = CONFIG.half_path/2
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
        poses_gt = np.zeros((xy_groundTruth.shape[1], 4, 4))
        poses_gt[:, :, :] = np.eye(4)
        poses_gt[:, :3, :3] = R_groundTruth
        half_N = CONFIG.num_robots // 2
        poses_gt[:, :2, 3] = xy_groundTruth.T + (robot_num-half_N)*CONFIG.half_path / CONFIG.num_robots / 5
        return poses_gt

    def generate_ground_truth_trajectories(self):
        #%%
        # Poses:
        poses_gt = list()
        for r in range(CONFIG.num_robots):
            pose_gt_per_robot = self.generate_ground_truth_trajectory(r)
            poses_gt.append(pose_gt_per_robot[None, ...])
        return np.concatenate(poses_gt, axis=0)

def is_orthogonal(Rs):
    return np.all(np.abs(batch_matmul(Rs, np.swapaxes(Rs, -1, -2)) - np.eye(3)) < 1e-4)

def is_det_one(Rs):
    n = Rs.size // 9
    return all((np.abs(np.linalg.det(Rs[i, :, :]) - 1) < 1e-4)
                for i in range(n))

def is_valid_so3(Rs):
    n = Rs.size // 9
    Rs = Rs.reshape(-1, 3, 3)
    return (is_orthogonal(Rs) and is_det_one(Rs))

def is_valid_se3(Ts):
    n = Ts.size // 16
    Rs = Ts[..., :3, :3]
    return (is_valid_so3(Rs)
            and np.all(np.abs(Ts[..., 3, :] - np.array([0, 0, 0, 1])) < 1e-4))

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
    noise1 = np.random.normal(0,CONFIG.sigma_pose_perturbation,(3,relative_poses.shape[0]))
    noise2 = np.random.normal(0,CONFIG.sigma_rotation_perturbation,(3,relative_poses.shape[0]))
    noise = np.append(noise1, noise2, axis=0)
    #noise[3:,:] = np.zeros((noise[3:,:].shape))
    noise_hat = hatMapR6(noise)
    noise_poses =  np.array(list(map(lambda i: expm(i), noise_hat)))
    perturbed_relative_poses = np.matmul(noise_poses, relative_poses)
    return perturbed_relative_poses

def plot_perturbed_relative_poses(xy_groundTruth, perturbed_relative_poses):
    # check plot of Perturbed Relative Poses
    xy = np.zeros((len(xy_groundTruth[0]),3))
    T = np.eye(4)
    T[0:2,3] = -xy_groundTruth[:,0]
    xy[0,:] = inv(T)[0:3,3]

    for i in np.arange(len(perturbed_relative_poses)):
        T = np.matmul(perturbed_relative_poses[i,:,:],T)
        assert is_valid_se3(T)
        xy[i+1,:] = inv(T)[0:3,3]


    plt.plot(xy[:,0],xy[:,1])
    plt.scatter(xy[0,0], xy[0,1])
    plt.title('perturbed path')
    plt.show()


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
    img_bounds_pxls = np.array([[CONFIG.img_shape[0]*0.02, CONFIG.img_shape[1]*0.02, 1],
                                [CONFIG.img_shape[0]*0.98, CONFIG.img_shape[1]*0.98, 1]])
    img_bounds_normed = np.linalg.solve(CONFIG.M_left[:3, :3], img_bounds_pxls.T).T
    x_lims = img_bounds_normed[:, 0]
    y_lims = img_bounds_normed[:, 1]
    z_lims = (0, # Point should be infront of camera BINGO!
                CONFIG.camera_limit) # Make it a realistic road with buildings etc.
    # Note that I reduced the camera range by 50/10 pixels from each edge to account for noise 
    # that will be added to generate the features.
    homo_features = np.append(features,np.ones((1,len(features[0]))),axis=0) # landmarks in normalized coordinates
    features_messages_positions = []
    for i in np.arange(len(poses_gt)):
        rel_features = CONFIG.opt_T_b @ invPoses_gt[i] @ homo_features #poses in camera frame (step i)
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


def plot_orientation(poses_gt):
    # %%
    # Plot of Orientation:
    # Inverted Rotating Matrices
    invR_groundTruth = np.array([R.T for R in poses_gt[0,:,:3,:3]])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-CONFIG.half_path, CONFIG.half_path])
    ax.set_ylim([-CONFIG.half_path, CONFIG.half_path])
    ax.plot(poses_gt[0, :, 0,3], poses_gt[0, :, 1,3])

    for i in np.arange(len(poses_gt[0, :, 0, 0]))[::3]:
        dxx, dzx, _ = poses_gt[0, i, :3, :3] @ np.array([1,0,0])*0.5
        plt.arrow(poses_gt[0,i,0,3], poses_gt[0,i,1,3], dxx, dzx, head_width=0.1)

        dxz, dzz, _ = poses_gt[0, i, :3, :3] @ np.array([0,1,0])*0.5
        plt.arrow(poses_gt[0,i, 0, 3], poses_gt[0,i,1,3],
                  dxz, dzz, head_width=0.1)
        #plt.pause(0.03)

    plt.title('orientation check')
    plt.show()

def plot_features_groundtruth(poses_gt, features):
    plt.plot(poses_gt[0, :, 0,3],poses_gt[0, :, 1,3])
    plt.scatter(features[0], features[1], s=3)
    plt.title('features ground truth')
    plt.show()


def generate_data(poses_gt):
    perturbed_relative_poses = [None]*CONFIG.num_robots
    for r in range(CONFIG.num_robots):
        perturbed_relative_poses[r] = compute_perturbed_relative_poses(
            poses_gt[r, ...])
        plot_perturbed_relative_poses(poses_gt[r, :, :2, 3].T,
                                      perturbed_relative_poses[r])

    #%%
    n = 1 # number of landmarks per robot per pose
    N = int(n*CONFIG.num_robots*len(poses_gt[0, ::2]))
    xy_mean = np.zeros(3) # Consider changing the y mean to somewhere above 0
    xy_sigma = np.diag([1,1,1])
    landmarks_distrib = np.random.multivariate_normal(xy_mean,xy_sigma,N)
    #features = landmarks_distrib + np.repeat(poses_gt[:,:3,3],n,axis=0) #landmarks positions in world frame
    features = landmarks_distrib + poses_gt[:, ::2,:3,3].reshape(-1, 3)
    features = features.T

    plot_features_groundtruth(poses_gt, features)
    features_messages_positions = [None]*CONFIG.num_robots
    for r in range(CONFIG.num_robots):
        features_messages_positions[r] = generate_feature_message_positions(features, poses_gt[r, ...])

    # Check Plot at pose k:
    fig, ax = plt.subplots(1,1)
    moviewriter = animation.FFMpegWriter(fps=15,
                                         metadata=dict(title='GT animation of robots'))
    with moviewriter.saving(fig, osp.join(CONFIG.save_path, 'generatedData.mp4'), 100):
        artist_list = []
        appearing_at_k = [None]*CONFIG.num_robots
        for k in range(len(features_messages_positions[0])):
            for r in range(CONFIG.num_robots):
                appearing_at_k[r] = list(features_messages_positions[r][k].keys())
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim(-2.0*CONFIG.half_path, 2.0*CONFIG.half_path)
            ax.set_ylim(-2.0*CONFIG.half_path, 2.00*CONFIG.half_path)
            artists = []
            for r, color in zip(range(CONFIG.num_robots), CONFIG.colors):

                artists.extend(
                    ax.plot(poses_gt[r, :, 0, 3].T,
                            poses_gt[r, :, 1, 3].T, color=color))
                artists.append(
                    ax.scatter(features[0,appearing_at_k[r]],
                            features[1,appearing_at_k[r]], s=10, color=color)
                )

                dxx, dzx, _ = poses_gt[r, k, :3, :3] @ np.array([0,1,0])*0.5
                artists.append(
                    ax.arrow(poses_gt[r, k, 0, 3].T,
                             poses_gt[r, k, 1, 3].T,
                             dxx, dzx,
                             joinstyle='miter',
                             linewidth=5,
                             color=color))
            artists.append(ax.set_title('features seen at time k = %d' % k))
            artist_list.append(artists)
            moviewriter.grab_frame()
            plt.pause(0.01)

    #### ArtistAnimation is so hard to make it work
    #anim = animation.ArtistAnimation(fig, artist_list, interval=50, repeat_delay=3000)
    #anim.save(osp.join(CONFIG.save_path, 'generatedData.mp4'))
    #plt.show()
    moviewriter.finish()


    #%%
    # Simulate Camera Features (u_left,v_left,u_right,v_right) with some perturbation
    # Noise Question: Should the noise 2D vector be the same for all features in one frame?
    # Assuming No (Independence for simplicity):
    invPoses_gt = [None]*CONFIG.num_robots
    for r in range(CONFIG.num_robots):
        invPoses_gt[r] = batch_invert_poses(poses_gt[r, ...])

    uv_mean = np.zeros(2 if CONFIG.use_monocular else 4 )
    features_messages = [[]]*CONFIG.num_robots
    for i in np.arange(len(poses_gt)):
        feats_poses = [None]*CONFIG.num_robots
        features_position_now = [None]*CONFIG.num_robots
        for r in range(CONFIG.num_robots):
            feats_poses[r] = features_messages_positions[r][i]
            features_position_now[r] = np.array(list(feats_poses[r].values())).T
            if features_position_now[r].size > 0:
                uv = stereo_obs_model(CONFIG.opt_T_b @ invPoses_gt[r][i],features_position_now[r], CONFIG.M)
                uv_noise = np.random.multivariate_normal(uv_mean, CONFIG.uv_sigma,uv.shape[1]).T
                uv_perturbed = uv + uv_noise
                uv_message = {list(feats_poses[r].keys())[x]:
                            uv_perturbed[:,x]
                            for x in range(len(list(feats_poses[r].keys())))}
                features_messages[r].append(uv_message)
            else:
                features_messages[r].append(dict())


    plot_orientation(poses_gt)

    #%%
    # Save Data:
    features_msg_file = osp.join(CONFIG.save_path, 'features_messages')
    visible_feat_file = osp.join(CONFIG.save_path, 'visible_features')
    gt_poses_file = osp.join(CONFIG.save_path, 'poses')
    inputs_file = osp.join(CONFIG.save_path, 'inputs')
    features_gt_file = osp.join(CONFIG.save_path, 'features_gt')

    # Store Features Massage:
    visible_feature_ids = dict()
    visible_feature_uvs = dict()
    for r, feat_messages in enumerate(features_messages):
        for t, fm_t in enumerate(feat_messages):
            fm_t_ids = np.array(list(fm_t.keys()))
            visible_feature_ids['id_%d_%d' % (r,t)] = fm_t_ids
            fm_t_uvs = (np.vstack(list(fm_t.values()))
                        if len(fm_t)
                        else np.zeros((0, 2 if CONFIG.use_monocular else 4)))
            visible_feature_uvs['uv_%d_%d' % (r, t)] = fm_t_uvs

    outfile = open(features_msg_file,'wb')
    pickle.dump(features_messages,outfile)
    outfile.close()
    
    np.savez(visible_feat_file,
             **visible_feature_ids,
             **visible_feature_uvs)
    
    # Store other stuff:
    np.savez(gt_poses_file, *[poses_gt[r] for r in range(CONFIG.num_robots)])
    np.savez(inputs_file, *perturbed_relative_poses)
    np.savez(features_gt_file, features)
    # %%

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

JSON_DUMP_CONFIG = dict(cls=NumpyJSONEncoder,
                        separators=(", ", ": "), indent=2)

def main():
    np.random.seed(CONFIG.seed)
    serializable_config = { k: v
                for k, v in vars(CONFIG).items()
                if not k.startswith('__') or not k.endswith('__')}
    json.dump(serializable_config,
              open(osp.join(CONFIG.save_path, 'config.json'), 'w'),
              **JSON_DUMP_CONFIG)
    #generate_data(EightPattern().generate_ground_truth_trajectories())
    generate_data(Lissajous().generate_ground_truth_trajectories())

if __name__ == '__main__':
    main()
