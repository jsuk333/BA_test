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

ORB_PARAMS = dict(
            nfeatures = 1000,
            scaleFactor = 2,
            nlevels = 3,
            edgeThreshold = 7,
            firstLevel = 0,
            WTA_K = 4,
            scoreType = cv2.ORB_HARRIS_SCORE,
            #scoreType = cv2.ORB_FAST_SCORE,
            patchSize = 50,
            fastThreshold = 5
        )


LK_PARAMS = dict( 
    winSize  = (50, 50),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10000,
    0.003)
) # criteria type, maxCount, epsilon

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

    """
    #print("result yaw", (result_yaw+initial_val[0]) * 180.0 / np.pi)
    #print("result tx",  result_tx)
    #print("result ty",  result_ty)

    #print("reverse_result yaw",     (reverse_result_yaw - initial_val[0]) * 180.0 / np.pi)
    #print("reverse_result tx",      reverse_result_tx)
    #print("reverse_result ty",      reverse_result_ty)
    """

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


# 번들 조정 함수 정의
def bundle_adjustment(points_2d, lidar_poses, observations):

    def residual(params):
        # 카메라 포즈와 3D 포인트 위치 추출
        num_lidars = len(lidar_poses) -1
        num_points = len(points_2d)
        lidar_params = params[:num_lidars * 3].reshape(num_lidars, 3)

        # to homogenous coordinate
        hm_points_2d = np.hstack((points_2d, np.ones((num_points,1)))).T
        
        # 잔차 계산
        residuals = []
        for i in range(num_lidars):
            reverse_R_i = yaw2mat(lidar_params[i, 0])
            reverse_t_i = lidar_params[i, 1:]

            translation_mat = np.array([
                [1.0,   0.0,    reverse_t_i[0]],
                [0.0,   1.0,    reverse_t_i[1]],
                [0.0,   0.0,    1.0],
            ])

            rotation_mat = np.array([
                [reverse_R_i[0,0], reverse_R_i[0,1], 0.0],
                [reverse_R_i[1,0], reverse_R_i[1,1], 0.0],
                [0.0,        0.0,        1.0],
            ])

            translated_points_2d = np.dot(translation_mat, hm_points_2d)
            projected_points = np.dot(rotation_mat, translated_points_2d).T
            residual_segs = []

            for observation, projected_point in zip(observations[i+1], projected_points[:,:2]):
                residual_seg = [0, 0]
                if not np.isnan(observation[0]):
                    residual_seg = (observation - projected_point).tolist()
                residual_segs.append(residual_seg)

            residual = np.array(residual_segs).ravel()
            residuals.append(residual)

        return np.concatenate(residuals)
    
    num_lidars = len(lidar_poses) -1

    # 초기 매개변수 추정
    #lidar_poses -= lidar_poses[0]
    initial_params = np.hstack([pose.ravel() for pose in lidar_poses[1:]])
    print("initial param", initial_params)
    
    # 번들 조정 수행
    result = least_squares(residual, initial_params, method = "lm", verbose=2)

    # 결과 반환
    optimized_lidar_poses = [lidar_poses[0].ravel()]
    for i in range(num_lidars):
        pose = result.x[i * 3:i*3 + 3]
        optimized_lidar_poses.append(pose)

    return optimized_lidar_poses

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
        [R[0,0], R[0,1], x],
        [R[1,0], R[1,1], y],
        [0.0,    0.0,    1.0],
    ])

    transform_mat = np.array([
        [R[0,0], R[0,1], 0.0,   x],
        [R[1,0], R[1,1], 0.0,   y],
        [0.0,    0.0,    0.0, 0.0],
        [0.0,    0.0,    0.0, 1.0],
    ])

    return transform_mat
"""
def get_relations(obs_points, obs_lidar_poses, crsp_data):
    f2i_idx_relations = {}
    f2f_relations = {}
    i2f_relations = {}
    f2i_matching_start = time.time()
    relation_id = 0

    obs_lidar_poses -= obs_lidar_poses[0]
    obs_tf_mats = []
    for obs_lidar_pose in obs_lidar_poses:
        yaw, x, y = obs_lidar_pose
        R_mat = yaw2mat(yaw)

        tf_mat = np.array([
            [R_mat[0,0], R_mat[0,1], x],
            [R_mat[1,0], R_mat[1,1], y],
            [0.0,        0.0,        1.0],
        ])

        obs_tf_mats.append(tf_mat)

    crsp_hash = crsp_data[relation_id]
    for matching_id, query_idx in enumerate(crsp_hash.keys()):
        train_idx = crsp_hash[query_idx]
        f2i_idx_relations.setdefault(matching_id, {})

        f2i_idx_relations[matching_id][0] = query_idx
        f2i_idx_relations[matching_id][1] = train_idx

    for relation_id in range(1, len(crsp_data)):
        crsp_hash = crsp_data[relation_id]
        matching_start = time.time()
        for matching_id in f2i_idx_relations.keys():
            query_image_idx = relation_id
            if query_image_idx not in f2i_idx_relations[matching_id].keys():
                continue
            query_idx = f2i_idx_relations[matching_id][query_image_idx]
            if query_idx not in crsp_hash.keys():
                continue
            train_image_idx = relation_id + 1
            f2i_idx_relations[matching_id][train_image_idx] = crsp_hash[query_idx]

    f2i_matching_end = time.time()
    print("f2i matching duration", f2i_matching_end - f2i_matching_start, "s")

    i2f_matching_start = time.time()
    fe_points = {}

    for match_idx in f2i_idx_relations.keys():
        for image_num in f2i_idx_relations[match_idx].keys():
            kep = obs_points[image_num]
            
            tf_mat = obs_tf_mats[image_num]
            key_idx = f2i_idx_relations[match_idx][image_num]
            x, y = kep[key_idx]
            fe_raw_point = np.array([x, y, 1]).reshape(-1, 3)
            fe_bundle_point = np.dot(tf_mat, fe_raw_point.T)
            fe_bundle_point = fe_bundle_point[:2]
            
            # based first image frame
            f2f_relations.setdefault(match_idx, {})
            f2f_relations[match_idx][image_num] = fe_bundle_point
            
            # based selected image frame
            i2f_relations.setdefault(image_num, {})
            i2f_relations[image_num][match_idx] = fe_raw_point[:2]

    i2f_matching_end = time.time()
    print("i2f matching duration", i2f_matching_end - i2f_matching_start, "s")

    return f2f_relations, i2f_relations
"""

def get_relations(obs_points, obs_lidar_poses, crsp_data):
    f2i_idx_relations = {}
    f2f_relations = {}
    i2f_relations = {}
    f2i_matching_start = time.time()
    relation_id = 0

    obs_lidar_poses -= obs_lidar_poses[0]
    obs_tf_mats = []
    for obs_lidar_pose in obs_lidar_poses:
        yaw, x, y = obs_lidar_pose
        R_mat = yaw2mat(yaw)

        tf_mat = np.array([
            [R_mat[0,0], R_mat[0,1], x],
            [R_mat[1,0], R_mat[1,1], y],
            [0.0,        0.0,        1.0],
        ])

        obs_tf_mats.append(tf_mat)

    crsp_hash = crsp_data[relation_id]
    for matching_id, query_idx in enumerate(crsp_hash.keys()):
        train_idx = crsp_hash[query_idx]
        f2i_idx_relations.setdefault(matching_id, {})

        f2i_idx_relations[matching_id][0] = query_idx
        f2i_idx_relations[matching_id][1] = train_idx

    for relation_id in range(1, len(crsp_data)):
        crsp_hash = crsp_data[relation_id]
        matching_start = time.time()
        for matching_id in f2i_idx_relations.keys():
            query_image_idx = relation_id
            if query_image_idx not in f2i_idx_relations[matching_id].keys():
                continue
            query_idx = f2i_idx_relations[matching_id][query_image_idx]
            if query_idx not in crsp_hash.keys():
                continue
            train_image_idx = relation_id + 1
            f2i_idx_relations[matching_id][train_image_idx] = crsp_hash[query_idx]

    f2i_matching_end = time.time()
    print("f2i matching duration", f2i_matching_end - f2i_matching_start, "s")

    i2f_matching_start = time.time()
    fe_points = {}

    for match_idx in f2i_idx_relations.keys():
        for image_num in f2i_idx_relations[match_idx].keys():
            try:
                kep = obs_points[image_num]
            except:
                print("image num", image_num, "match_idx", match_idx)
                raise Exception("!!!!")
            
            tf_mat = obs_tf_mats[image_num]
            key_idx = f2i_idx_relations[match_idx][image_num]
            x, y = kep[key_idx]
            fe_raw_point = np.array([x, y, 1]).reshape(-1, 3)
            fe_bundle_point = np.dot(tf_mat, fe_raw_point.T)
            fe_bundle_point = fe_bundle_point[:2]
            
            # based first image frame
            f2f_relations.setdefault(match_idx, {})
            f2f_relations[match_idx][image_num] = fe_bundle_point
            
            # based selected image frame
            i2f_relations.setdefault(image_num, {})
            i2f_relations[image_num][match_idx] = fe_raw_point[:2]

    i2f_matching_end = time.time()
    print("i2f matching duration", i2f_matching_end - i2f_matching_start, "s")

    return f2f_relations, i2f_relations


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
    rigid_transform[:2,:2] = ransac.estimator_.coef_[:4].reshape((2,2))
    rigid_transform[:2, 3] = ransac.estimator_.intercept_[:2]
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

def main():
    filenames = os.listdir()
    filenames = sorted(filenames)
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]
    res = 0.125

    odom_traj = load_trajectory("odom_traj.txt")
    odom_traj[:,1:] -= odom_traj[0,1:]
    initial_heading = odom_traj[0][0]
    initial_R_mat = yaw2mat(initial_heading)
    image_heading = 0.0
    heading = 0.0
    start_idx = 450
    prev = extract_feature(filenames[start_idx])
    obs_lidar_poses = [[image_heading, 0.0,0.0]]
    global_lidar_poses = [[heading, 0.0,0.0]]

    prev_obs_fe_points_set = []
    curr_obs_fe_points_set = []

    obs_fe_points = []
    images = [prev]
    prev_view_image = prev.copy()
    view_images = [prev_view_image]
    print("start heading", odom_traj[0] * 180.0/np.pi)

    prev_indices = np.transpose(np.nonzero(prev)) - 512 
    source_points = prev_indices

    obs_points = [source_points]

    # viewer setting
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    end_flag = False
    colors = ["gainsboro", "silver", "darkgrey", "grey", "dimgrey"]
    
    # bundel adjustment set
    bundle_num = 5
    skip_num = 0
    match_num = bundle_num - skip_num
    keyPoint_table = []
    f2i_idx_relations = {}

    # extract initial feature
    cloud_tf = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, 0.0, 1.0]
            ]
    )

    clouds = []
    # main loop
    opt_lidar_locs = []
    crsp_data = []
    match_data = []
    for image_idx in range(start_idx+1, len(filenames)):
        loop_start_time = time.time()
        filename = filenames[image_idx]
        print("filename", filename)

        ############################# image load #####################################
        # image load
        curr = extract_feature(filename)
        images.append(curr.copy())
        view_images.append(curr.copy())
        last_obs_images = images[-bundle_num:]
        ##############################################################################

        ############################# icp ####################################
        # 초기 변환 매개변수 설정 (회전과 이동 모두 고려)
        source_points = obs_points[-1]
        curr_indices = np.transpose(np.nonzero(curr)) - 512
        target_points = curr_indices
        obs_points.append(target_points)
        ##############################################################################

        ############################# odom pose #####################################
        diff_odom_traj = odom_traj[image_idx] - odom_traj[image_idx-1]
        #print(odom_traj[image_idx])
        #print(odom_traj[image_idx-1])
        last_odom_traj = odom_traj[image_idx+1-bundle_num:image_idx+1] - odom_traj[image_idx+1-bundle_num]
        last_odom_traj[:,1:] /=0.125
        dx = -last_odom_traj[:,2].copy()
        dy = last_odom_traj[:,1].copy()
        last_odom_traj[:,1] = dx
        last_odom_traj[:,2] = dy
        print("start_idx", image_idx+1-bundle_num, "image_idx", image_idx)
        print("last odom traj", last_odom_traj)

        ############################# lidar pose #####################################


        aprx_pose_est_start_time = time.time()
        odom_yaw = diff_odom_traj[0]
        odom_dx  = -diff_odom_traj[2]/res
        odom_dy  = diff_odom_traj[1]/res
        crsp_hash, crsp_set, rigid_transform = fe_matching(source_points, target_points,(odom_yaw, odom_dx, odom_dy), one2one = True)
        print("########################################################################################### 1st crsp_hash size", len(crsp_hash))
        aprx_pose_est_end_time = time.time()

        # to homogenous coordinate
        hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
        transformed_source_points = np.dot(rigid_transform, hm_source_points).T

        icp_yaw = -mat2yaw(rigid_transform[:2,:2])
        icp_dx =  rigid_transform[0,3]
        icp_dy =  rigid_transform[1,3]

        print("icp param yaw, dx, dy", icp_yaw, icp_dx, icp_dy)

        #crsp_hash, crsp_set, rigid_transform = fe_matching(source_points, target_points,(yaw, dx, dy), one2one=True)
        #crsp_hash = first_crsp_hash 
        #crsp_set   = first_crsp_set
        #rigid_transform = first_transform

        #print("########################################################################################### 2nd crsp_hash size", len(crsp_hash))

        crsp_data.append(crsp_hash)
        crsp_source_points = source_points[crsp_set[:,0].ravel()]
        crsp_target_points = target_points[crsp_set[:,1].ravel()]

        ########################################################## relation view ####################################
        """
        delta_x = 512
        delta_y = 512
        # to homogenous coordinate
        hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
        transformed_source_points = np.dot(rigid_transform, hm_source_points).T

        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(curr)

        plt.subplot(1,2,2)
        plt.imshow(prev+curr)

        for crsp_source_point, crsp_target_point in zip(crsp_source_points, crsp_target_points):
            plt.plot([crsp_source_point[1]+delta_x, crsp_target_point[1]+delta_x], [crsp_source_point[0]+delta_y, crsp_target_point[0]+delta_y])

        plt.show()
        """
        #############################################################################################################

        yaw = mat2yaw(rigid_transform[:2,:2])
        dx =          rigid_transform[0,3]
        dy =          rigid_transform[1,3]


        image_heading -= yaw
        heading -= yaw
        R_mat = yaw2mat(image_heading)
        dt = np.dot(R_mat, np.array([dx, dy]))
        print("dx", dx*0.125, "dy", dy*0.125, "yaw", yaw*180.0/np.pi, "heading", heading*180.0/np.pi)
        
        # predicted curr lidar pose based 0th lidar frame
        obs_lidar_pose = [image_heading, obs_lidar_poses[-1][1] + dt[0], obs_lidar_poses[-1][2] + dt[1]]
        obs_lidar_poses.append(obs_lidar_pose)
        last_obs_lidar_poses = np.array(obs_lidar_poses[-bundle_num:])
        last_obs_points = obs_points[-bundle_num:]
        last_crsp_data = crsp_data[-(bundle_num -1):]

        prev = curr
        match_std_li = []
        ############################## relation set ##################################

        relation_start_time = time.time()
        f2f_relations, i2f_relations = get_relations(last_obs_points, last_obs_lidar_poses, last_crsp_data)
        relation_end_time = time.time()
        print("relation duration ", relation_end_time - relation_start_time, "s")

        fe_enroll_start_time = time.time()
        color_index = 0
        points_2d = []
        for match_id in sorted(f2f_relations.keys()):
            match_cluster = []
            if len(f2f_relations[match_id].keys()) < match_num:
                continue

            points_2d.append(f2f_relations[match_id][0])

            match_cluster = []
            if bundle_num != len(f2f_relations[match_id].keys()):
                continue
            
            for image_num in range(bundle_num):
                fe_raw_point = f2f_relations[match_id][image_num]
                match_cluster.append(fe_raw_point+512)

            match_cluster = np.array(match_cluster)
            match_data.append(match_cluster)


        fe_enroll_end_time = time.time()
        print("feature enroll duration ", fe_enroll_end_time - fe_enroll_start_time, "s")

        print("Relation count :", len(points_2d))
        #if len(points_2d) < 10:
        #    continue

        ##############################################################################
        if len(last_obs_images) < bundle_num:
            continue

        match_std_array = np.array(match_std_li)

        #plt.show()

        ############################# estimate   #####################################

        # vector space to affine space
        points_2d = np.array(points_2d).reshape(-1,2)

        # get observation(raw sensor data)
        observations = []
        for image_num in range(bundle_num):
            observation = []
            
            for match_id in sorted(f2f_relations.keys()):
                if len(f2f_relations[match_id].keys()) < match_num:
                    continue
                if image_num in f2f_relations[match_id].keys():
                    observation.append(i2f_relations[image_num][match_id].ravel()[:2])
                else:
                    observation.append([np.nan, np.nan])

            observation = np.array(observation)
            observations.append(observation)

        # do bundle adjustment
        bundle_adjustment_start_time = time.time()
        opt_lidar_poses = bundle_adjustment(points_2d, last_obs_lidar_poses, observations)
        #opt_lidar_poses = obs_lidar_poses
        bundle_adjustment_end_time = time.time()
        print("bundle_adjustment duration ", bundle_adjustment_end_time - bundle_adjustment_start_time, "s")

        # get optimized lidar pose
        for image_num, (opt_lidar_pose, obs_lidar_pose) in enumerate(zip(opt_lidar_poses, last_obs_lidar_poses)):
        #for image_num, (opt_lidar_pose, obs_lidar_pose) in enumerate(zip(obs_lidar_poses, last_obs_lidar_poses)):
            image_yaw = opt_lidar_pose[0]
            obs_yaw = obs_lidar_pose[0]
            image_position = opt_lidar_pose[1:]
            obs_position = obs_lidar_pose[1:]
            #image_position = np.dot(initial_R_mat, image_position)
            #print("bundle_tvec", image_pose*0.125, "bundle_yaw", image_yaw*180.0/np.pi)

            #for fe_point in points_2d:
            #    plt.plot([obs_position[0], fe_point[0]], [obs_position[1], fe_point[1]], c = colors[image_num])

            #for fe_point in opt_points_2d:
            #    plt.plot([image_position[0], fe_point[0]], [image_position[1], fe_point[1]], c = colors[image_num])

            if image_num < bundle_num-1:
                opt_lidar_locs.append([image_yaw, image_position[0], image_position[1]])
            if image_num == bundle_num -1:
                opt_lidar_locs.append([image_yaw, image_position[0], image_position[1]])

        # draw optimized lidar trajectory
        print("index", image_idx)
        opt_lidar_locs = np.array(opt_lidar_locs)
        #plt.plot(trajectory[:,0], trajectory[:,1], "r")

        if image_idx > start_idx + bundle_num-3:
            break

        print("______________________________________________________________")

    obs_lidar_poses = np.array(obs_lidar_poses)

    #o3d.visualization.draw_geometries(clouds)
    
    image2real_res = 0.125
    real2image_res = 1
    start = 0

    # draw trajectory

    # draw trajectory - odom
    plt.figure()
    plt.plot(    odom_traj[start:,1]/real2image_res, odom_traj[start:,2]/real2image_res, "g")
    plt.scatter( odom_traj[start:,1]/real2image_res, odom_traj[start:,2]/real2image_res)
    
    # draw trajectory - observation
    plt.plot(    -obs_lidar_poses[-bundle_num:,2]*image2real_res, -obs_lidar_poses[-bundle_num:,1]*image2real_res,  c="r")
    plt.scatter( -obs_lidar_poses[-bundle_num:,2]*image2real_res, -obs_lidar_poses[-bundle_num:,1]*image2real_res,  c="r")

    # draw trajectory - bundle adjust
    plt.plot(   -opt_lidar_locs[:,2]*image2real_res, -opt_lidar_locs[:,1]*image2real_res, 'b')
    plt.scatter(-opt_lidar_locs[:,2]*image2real_res, -opt_lidar_locs[:,1]*image2real_res, c='b')
    #print(odom_traj[:bundle_num,0])

    plt.scatter(points_2d[:,0], points_2d[:,1], s = 1)
    plt.axis("equal")

    # draw stiching image
    last_view_images = view_images[-bundle_num:]
    result  = last_view_images[0].copy()
    result2 = last_view_images[0].copy()
    print("-----------------------------------------")
    print("last obs poses", last_obs_lidar_poses)
    print("-----------------------------------------")
    print("opt_lidar_locs", opt_lidar_locs)
    print("-----------------------------------------")
    opt_lidar_locs[:,0] -= opt_lidar_locs[0,0]
    for i, image in enumerate(last_view_images[1:]):
        # traj
        theta = last_obs_lidar_poses[i,0]/np.pi*180
        tx = last_obs_lidar_poses[i,1]
        ty = last_obs_lidar_poses[i,2]

        # 이미지의 중심을 계산
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전을 위한 변환 행렬 생성
        rotation_angle = theta  # 회전 각도
        scale_factor = 1.0   # 스케일 인자

        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, scale_factor)
        #print("odom", odom_traj[i,0]/np.pi*180.0, odom_traj[i,1], odom_traj[i,2])

        # 회전 및 이동 적용
        rotation_matrix[0,2] -= ty
        rotation_matrix[1,2] -= tx
        warp = cv2.warpAffine(image, rotation_matrix, (w, h))

        result += warp

        # bundle adjust
        theta = opt_lidar_locs[i,0]/np.pi*180
        tx    = opt_lidar_locs[i,1]
        ty    = opt_lidar_locs[i,2]

        rotation_angle = theta  # 회전 각도

        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, scale_factor)
        #print("odom", odom_traj[i,0], odom_traj[i,1], odom_traj[i,2])

        # 회전 및 이동 적용
        rotation_matrix[0,2] -= ty
        rotation_matrix[1,2] -= tx
        warp2 = cv2.warpAffine(image, rotation_matrix, (w, h))

        result2 += warp2


    # fix me
    plt.figure()
    plt.title("observation merge")
    plt.imshow(result)

    plt.figure()
    plt.title("bundle adjust merge")
    plt.imshow(result2)

    plt.axis("equal")

    plt.show()
    ###############################################################################

if __name__ == "__main__":
    main()
