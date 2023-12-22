import os
import sys
import time
import numpy as np
import cv2
from sklearn import linear_model
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import open3d as o3d
import random


BLUR_SIZE = (7,7)
BLUR_VAR = 7

def icp(source_points, target_points, ):
    max_iterations=50
    tolerance=1e-4
    """
    Iterative Closest Point (ICP) algorithm for point cloud registration.

    Args:
    source_points (numpy.ndarray): Source point cloud (Nx3).
    target_points (numpy.ndarray): Target point cloud (Nx3).
    max_iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    R (numpy.ndarray): Rotation matrix (3x3).
    t (numpy.ndarray): Translation vector (3x1).
    """
    R = np.eye(3)
    t = np.zeros((3, 1))

    # ICP를 사용하여 소스 클라우드를 타겟 클라우드에 정합시킵니다.
    result = o3d.cuda.pybind.pipelines.registration.registration_generalized_icp(
            source_cloud, target_cloud, 10, trans_init)

    reverse_result = o3d.cuda.pybind.pipelines.registration.registration_generalized_icp(
            target_cloud, source_cloud, 10, trans_init)

    result_yaw = mat2yaw(result.transformation[:2,:2])
    result_tx = result.transformation[2,0]
    result_ty = result.transformation[2,1]

    
    for i in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(target_points)
        distances, indices = tree.query(source_points)
        
        # Filter correspondences based on distance threshold
        correspondences = source_points[distances < tolerance]
        target_correspondences = target_points[indices[distances < tolerance]]
        
        if len(correspondences) == 0:
            break
        
        # Compute optimal rotation and translation using Procrustes analysis
        R_i, t_i = orthogonal_procrustes(target_correspondences.T, correspondences.T)
        
        # Update transformation
        R = R_i.dot(R)
        t = R_i.dot(t) + t_i.reshape(3, 1)
        
        # Check for convergence
        if np.allclose(R_i, np.eye(3), atol=tolerance) and np.allclose(t_i, np.zeros((3, 1)), atol=tolerance):
            break

    return R, t


def fe_matching(source_points, target_points, initial_val, one2one = False):
    ############################### convert points to cloud #############################
    tf_mat = get_tf(initial_val)
    hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
    transformed_source_points = np.dot(tf_mat, hm_source_points).T[:,:2]

    source_cloud = points2cloud(transformed_source_points)
    target_cloud = points2cloud(target_points)
    ######################################################################################

    trans_init = np.asarray(
            [
                [1.0, 0.0, 0.0  , 0.0],
                [0.0, 1.0, 0.0  , 0.0],
                [0.0, 0.0, 1.0  , 0.0], 
                [0.0, 0.0, 0.0  , 1.0]])


    # ICP를 사용하여 소스 클라우드를 타겟 클라우드에 정합시킵니다.
    #result = o3d.pipelines.registration.registration_icp(    source_cloud, target_cloud, 2, trans_init,    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #reverse_result = o3d.pipelines.registration.registration_icp(    target_cloud, source_cloud, 2, trans_init,    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    result = o3d.cuda.pybind.pipelines.registration.registration_generalized_icp(
            source_cloud, target_cloud, 10, trans_init)

    reverse_result = o3d.cuda.pybind.pipelines.registration.registration_generalized_icp(
            target_cloud, source_cloud, 10, trans_init)

    result_yaw = mat2yaw(result.transformation[:2,:2])
    result_tx = result.transformation[2,0]
    result_ty = result.transformation[2,1]

    reverse_result_yaw = mat2yaw(reverse_result.transformation[:2,:2])
    reverse_result_tx = reverse_result.transformation[2,0]
    reverse_result_ty = reverse_result.transformation[2,1]
    print("______________________________________________________________________________")
    print("result yaw", (result_yaw+initial_val[0]))
    #print("result yaw", (result_yaw+initial_val[0]) * 180.0 / np.pi)
    print("result tx",  result_tx)
    print("result ty",  result_ty)

    #print("reverse_result yaw",     (reverse_result_yaw - initial_val[0]) * 180.0 / np.pi)
    print("reverse_result yaw",     (reverse_result_yaw - initial_val[0]))
    print("reverse_result tx",      reverse_result_tx)
    print("reverse_result ty",      reverse_result_ty)

    # 매칭된 포인트의 인덱스 확인
    forward_crsp_set = np.asarray(result.correspondence_set)
    backward_crsp_set = np.asarray(reverse_result.correspondence_set)

    # 1 vs 1 correspondence
    aprx_pose_est_start_time = time.time()

    forward_crsp_hash = {}
    crsp_hash = {}
    crsp_set = []
    for i in range(len(forward_crsp_set)):
        key, val = forward_crsp_set[i]
        forward_crsp_hash[key] = val

    for i in range(len(backward_crsp_set)):
        key, val = backward_crsp_set[i]
        
        if val not in forward_crsp_hash.keys():
            continue

        if key != forward_crsp_hash[val]:
            continue

        crsp_hash[val] = key
        crsp_set.append([val, key])

    if one2one:
        crsp_set = np.array(crsp_set)
    else:
        crsp_set = forward_crsp_set
        crsp_hash = forward_crsp_hash

    #crsp_data.append(crsp_hash)
    crsp_source_points = source_points[crsp_set[:,0].ravel()]
    crsp_target_points = target_points[crsp_set[:,1].ravel()]
    rigid_transform = pose_estimate(crsp_source_points, crsp_target_points)

    aprx_pose_est_end_time = time.time()
    print("approximately pose estimation duration ", aprx_pose_est_end_time - aprx_pose_est_start_time, "s")

    return crsp_hash, crsp_set, rigid_transform

def yaw2mat(angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
            ])
    return R

def mat2yaw(R):
    yaw = np.arctan2(-R[1,0], R[0,0])
    return yaw

def get_tf(pose):
    yaw, x, y = pose
    R = yaw2mat(yaw)
    transform_mat = np.array([
        [R[0,0], R[0,1], 0.0,   x],
        [R[1,0], R[1,1], 0.0,   y],
        [0.0,    0.0,    0.0, 0.0],
        [0.0,    0.0,    0.0, 1.0],
    ])

    return transform_mat

def points2cloud(points, color_coef = 4.0):

    hm_points = np.hstack((points, np.ones((points.shape[0],1)))).T
    cloud = o3d.geometry.PointCloud()
 
    cloud.points = o3d.utility.Vector3dVector(hm_points.T.astype(np.float))
 
    color = np.ones_like(hm_points.T)/ color_coef
    cloud.colors = o3d.utility.Vector3dVector(color)
    return cloud

def pose_estimate(crsp_source_points, crsp_target_points):
    ransac = RANSACRegressor()

    # RANSAC 모델에 데이터를 적합시킴 (회전과 이동을 포함하는 Rigid Transform 모델 추정)
    ransac.fit(crsp_source_points, crsp_target_points)

    # 추정된 Rigid Transform 행렬 출력
    rigid_transform = np.eye(4)
    rigid_transform[:2,:2] = ransac.estimator_.coef_
    rigid_transform[:2, 3] = ransac.estimator_.intercept_
    return rigid_transform

def extract_feature(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, BLUR_SIZE, BLUR_VAR)
    canny_threshold_low =50
    canny_threshold_high = 110
    edge = cv2.Canny(img, canny_threshold_low, canny_threshold_high)
    img += edge
    img = cv2.GaussianBlur(img, BLUR_SIZE, BLUR_VAR)

    return img

def load_trajectory(filename):

    trajectory = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(" ")
            timestamp = float(tokens[0])
            yaw = float(tokens[1])/180.0 *np.pi# + np.pi/2
            x = float(tokens[2])
            y = float(tokens[3])
            z = float(tokens[4])
            trajectory.append([yaw, x, y])

    return np.array(trajectory)

def main():

    first_idx = int(sys.argv[1])
    second_idx = int(sys.argv[2])

    res = 0.125
    
    filenames = os.listdir()
    filenames = sorted(filenames)
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]

    odom_traj = load_trajectory("odom_traj.txt")

    prev_filename = filenames[first_idx]
    prev = extract_feature(prev_filename)
    prev_indices = np.transpose(np.nonzero(prev)) - 512 
    source_points = prev_indices

    colors = ["gainsboro", "silver", "darkgrey", "grey", "dimgrey"]
 
    curr_filename = filenames[second_idx]
    curr = extract_feature(curr_filename)

    curr_indices = np.transpose(np.nonzero(curr)) - 512 
    target_points = curr_indices

    diff_odom_traj = odom_traj[second_idx] - odom_traj[first_idx]
    #yaw = diff_odom_traj[0] 
    #dx  = -diff_odom_traj[2]/res
    #dy  = diff_odom_traj[1]/res

    yaw = diff_odom_traj[0] 
    dx  = -diff_odom_traj[2]/res
    dy  = diff_odom_traj[1]/res     
    odom_tf_mat = get_tf((yaw, dx, dy))
    hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
    odom_source_points = np.dot(odom_tf_mat, hm_source_points).T[:,:2]

    print("odom dtheta" , yaw)
    print("odom dx"     , dx)
    print("odom dy"     , dy)

    aprx_pose_est_start_time = time.time()
    crsp_hash, crsp_set, rigid_transform = fe_matching(source_points, target_points,(yaw, dx, dy), one2one=True)
    print("1st crsp_hash size", len(crsp_hash))
    icp_first_source_points = np.dot(rigid_transform, hm_source_points).T[:,:2]

    icp_yaw = mat2yaw(rigid_transform[:2,:2])
    icp_dx  = rigid_transform[0,3]
    icp_dy  = rigid_transform[1,3]

    print("icp dtheta"  , icp_yaw)
    print("icp dx"      , icp_dx)
    print("icp dy"      , icp_dy)


    #crsp_hash, crsp_set, rigid_transform = fe_matching(source_points, target_points,(icp_yaw, icp_dx, icp_dy), one2one=True)
    #print("2nd crsp_hash size", len(crsp_hash))
    #crsp_source_points = source_points[crsp_set[:,0].ravel()]
    #crsp_target_points = target_points[crsp_set[:,1].ravel()]
    #icp_second_source_points = np.dot(rigid_transform, hm_source_points).T[:,:2]

    #icp_yaw = mat2yaw(rigid_transform[:2,:2])
    #icp_dx  = rigid_transform[0,3]
    #icp_dy  = rigid_transform[1,3]

    #print("icp dtheta"  , icp_yaw)
    #print("icp dx"      , icp_dx)
    #print("icp dy"      , icp_dy)


    delta_x = 512
    delta_y = 512
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(curr)
    
    plt.subplot(1,2,2)
    x_idx = 1
    y_idx = 0
    #plt.imshow(prev+curr)
    #plt.scatter(source_points[:,x_idx]      + delta_x , source_points[:,y_idx]      + delta_y,s = 6,  c = "r", label = "prev")
    plt.scatter(odom_source_points[:,x_idx] + delta_x , odom_source_points[:,y_idx] + delta_y,s = 4,  c = "g", label = "odom")
    plt.scatter(icp_first_source_points[:,x_idx]  + delta_x , icp_first_source_points[:,y_idx]  + delta_y,s = 1,  c = "r", label = "icp")
    plt.scatter(target_points[:,x_idx]      + delta_x , target_points[:,y_idx]      + delta_y,s = 2,  c = "b", label = "curr")
    #plt.scatter(icp_second_source_points[:,x_idx]  + delta_x , icp_second_source_points[:,y_idx]  + delta_y,s = 1,  c = "c", label = "icp")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.legend()

    #for crsp_source_point, crsp_target_point in zip(crsp_source_points, crsp_target_points):
    #    plt.plot([crsp_source_point[x_idx] + delta_x, crsp_target_point[x_idx] + delta_x], [crsp_source_point[y_idx] + delta_y, crsp_target_point[y_idx] + delta_y])

    plt.show()

if __name__ == "__main__":
    main()
