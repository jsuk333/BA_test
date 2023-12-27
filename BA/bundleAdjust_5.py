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
BLUR_VAR = 3

BUNDLE_NUM = 5
SKIP_NUM = 0
MATCH_NUM = BUNDLE_NUM - SKIP_NUM

START_IDX = 130


def fe_matching(source_points, target_points, initial_val, one2one = False):
    ############################### convert points to cloud #############################
    odom_tf = get_tf(initial_val)
    hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
    odom_source_points = np.dot(odom_tf, hm_source_points).T[:,:2]


    source_cloud = points2cloud(odom_source_points)
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

    crsp_source_points = source_points[crsp_set[:,0].ravel()]
    crsp_target_points = target_points[crsp_set[:,1].ravel()]
    rigid_transform = pose_estimate(crsp_source_points, crsp_target_points)

    aprx_pose_est_end_time = time.time()
    print("initial val", initial_val[0], initial_val[1], initial_val[2])

    yaw = mat2yaw(rigid_transform[:2,:2])
    du =  rigid_transform[0,3]
    dv =  rigid_transform[1,3]

    print("rigid_transform", yaw , du, dv)

    icp_source_points = np.dot(rigid_transform, hm_source_points).T

    #plt.scatter(odom_source_points[:,0], odom_source_points[:,1], c = "r", label = "odom")
    #plt.scatter(icp_source_points[:,0], icp_source_points[:,1], c = "g", label = "icp")
    #plt.scatter(target_points[:,0], target_points[:,1], c = "b", label = "target")
    #plt.axis("equal")
    #plt.legend()
    #plt.show()


    return crsp_hash, crsp_set, rigid_transform


# 번들 조정 함수 정의
def bundle_adjustment(points_2d, lidar_poses, observations):

    def residual(params):
        # 카메라 포즈와 3D 포인트 위치 추출
        num_lidars = len(lidar_poses)
        num_points = len(points_2d)
        lidar_params = params[:num_lidars * 3].reshape(num_lidars, 3)
        point_params = params[num_lidars * 3:].reshape(num_points, 2)

        # to homogenous coordinate
        hm_points_2d = np.hstack((point_params, np.ones((num_points,1)))).T
        
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

            for observation, projected_point in zip(observations[i], projected_points[:,:2]):
                residual_seg = [0, 0]
                if not np.isnan(observation[0]):
                    residual_seg = (observation - projected_point).tolist()
                residual_segs.append(residual_seg)

            residual = np.array(residual_segs).ravel()
            residuals.append(residual)

        return np.concatenate(residuals)
    
    num_lidars = len(lidar_poses)
    num_points = len(points_2d)

    # 초기 매개변수 추정
    #lidar_poses -= lidar_poses[0]
    print(lidar_poses)
    initial_params = np.hstack([pose.ravel() for pose in lidar_poses] + [point_2d.ravel() for point_2d in points_2d])
    
    # 번들 조정 수행
    result = least_squares(residual, initial_params, method = "lm", verbose=2)

    # 결과 반환
    #optimized_lidar_poses = []
    #for i in range(num_lidars):
    #    pose = result.x[i * 3:i*3 + 3]
    #    optimized_lidar_poses.append(pose)


    optimized_lidar_poses = result.x[:num_lidars * 3].reshape(num_lidars, 3)
    optimized_points_2d   = result.x[num_lidars * 3:].reshape(num_points, 2)

    return optimized_lidar_poses, optimized_points_2d

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

def get_relations(obs_points, obs_lidar_poses, crsp_hash_bundle):
    f2i_idx_relations = {}
    f2f_relations = {}
    i2f_relations = {}
    f2i_matching_start = time.time()

    obs_lidar_poses -= obs_lidar_poses[0]
    obs_tf_mats = []
    for obs_lidar_pose in obs_lidar_poses:
        yaw, x, y = obs_lidar_pose
        R_mat = yaw2mat(-yaw)

        tf_mat = np.array([
            [R_mat[0,0], R_mat[0,1], -x],
            [R_mat[1,0], R_mat[1,1], -y],
            [0.0,        0.0,        1.0],
        ])

        obs_tf_mats.append(tf_mat)

    i2i_relations = {}
    match_idx_count = 0
    for query_img_idx in crsp_hash_bundle.keys():
        for train_img_idx in crsp_hash_bundle[query_img_idx].keys():
            crsp_hash = crsp_hash_bundle[query_img_idx][train_img_idx]
            for query_idx in crsp_hash.keys():
                train_idx = crsp_hash[query_idx]
                i2i_relations.setdefault(query_img_idx, {})
                i2i_relations.setdefault(train_img_idx, {})
                match_idx = match_idx_count
                
                if query_idx in i2i_relations[query_img_idx].keys() and train_idx in i2i_relations[train_img_idx].keys():
                    continue
                elif query_idx in i2i_relations[query_img_idx].keys():
                    match_idx = i2i_relations[query_img_idx][query_idx]
                elif train_idx in i2i_relations[train_img_idx].keys():
                    match_idx = i2i_relations[train_img_idx][train_idx]
                else:
                    match_idx = match_idx_count
                    match_idx_count += 1

                i2i_relations[query_img_idx][query_idx] = match_idx
                i2i_relations[train_img_idx][train_idx] = match_idx
                f2i_idx_relations.setdefault(match_idx, {})
                f2i_idx_relations[match_idx][query_img_idx] = query_idx
                f2i_idx_relations[match_idx][train_img_idx] = train_idx

    f2i_matching_end = time.time()

    #######################################################################################################

    i2f_matching_start = time.time()
    fe_points = {}
    #print(obs_tf_mats)

    for match_idx in f2i_idx_relations.keys():
        print("match idx ", match_idx)
        for image_num in sorted(f2i_idx_relations[match_idx].keys()):
            try:
                kep = obs_points[image_num]
            except:
                raise Exception("!!!!")
            
            tf_mat = obs_tf_mats[image_num]
            key_idx = f2i_idx_relations[match_idx][image_num]
            try:
                x, y = kep[key_idx]
            except:
                raise Exception("points len", len(kep), "key_idx", key_idx, "image_num", image_num)

            fe_raw_point = np.array([x, y, 1]).reshape(3, 1)
            dx = tf_mat[0,2]
            dy = tf_mat[1,2]
            translation_mat = np.array([
                [1.0,   0.0,    dx],
                [0.0,   1.0,    dy],
                [0.0,   0.0,    1.0],
            ])

            rotation_mat = np.array([
                [tf_mat[0,0], -tf_mat[0,1], 0.0],
                [-tf_mat[1,0], tf_mat[1,1], 0.0],
                [0.0,        0.0,        1.0],
            ])
            translated_point = np.dot(translation_mat, fe_raw_point)
            #test_point = np.dot(tf_mat, fe_raw_point)
            #print("-----------------------------------------------------")
            #print(test_point)
            fe_bundle_point = np.dot(rotation_mat, translated_point)
            print("        ", "fe matching dx, dy", dx, dy)
            print("        ", fe_raw_point.ravel(), " ", fe_bundle_point.ravel())
            #print(fe_bundle_point)
            fe_bundle_point = fe_bundle_point.ravel()[:2]
            #print("fe_bundle_point", fe_bundle_point)
            # based selected_match_idx
            f2f_relations.setdefault(match_idx, {})
            f2f_relations[match_idx][image_num] = fe_bundle_point
            f2f_relations[match_idx][image_num] = np.array([x,y])
            #print(match_idx, fe_raw_point, fe_bundle_point)
            
            # based selected image frame
            i2f_relations.setdefault(image_num, {})
            i2f_relations[image_num][match_idx] = fe_raw_point[:2]

    i2f_matching_end = time.time()

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
    filenames = os.listdir("img")
    filenames = sorted(filenames)
    filenames = [os.path.join("img", filename)  for filename in filenames if filename[:9] == "mat_alt_0"]
    res = 0.125

    odom_traj = load_trajectory("odom_traj.txt")
    odom_traj[:,1:] -= odom_traj[0,1:]
    initial_heading = odom_traj[0][0]
    initial_R_mat = yaw2mat(initial_heading)
    image_heading = -initial_heading
    heading = 0.0
    obs_lidar_poses = [[image_heading, 0.0,0.0]]

    prev = extract_feature(filenames[START_IDX])
    images = [prev]
    prev_view_image = prev.copy()
    view_images = [prev_view_image]
    prev_indices = np.transpose(np.nonzero(prev)) - 512 
    source_points = prev_indices
    obs_points = [source_points]

    print("start heading", odom_traj[0] * 180.0/np.pi)

    # viewer setting
    end_flag = False
    colors = ["gainsboro", "silver", "darkgrey", "grey", "dimgrey"]
    
    # extract initial feature
    cloud_tf = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, 0.0, 1.0]
            ]
    )

    # main loop
    opt_lidar_locs = []
    crsp_hash_data = {}
    match_data = []
    crsp_hash_bundle  = {}
    for image_idx in range(START_IDX+1, len(filenames)):
        loop_start_time = time.time()
        filename = filenames[image_idx]
        print("filename", filename)

        ############################# image load #####################################
        # image load
        curr = extract_feature(filename)
        images.append(curr.copy())
        view_images.append(curr.copy())
        last_obs_images = images[-BUNDLE_NUM:]
        ##############################################################################

        ############################# icp ####################################
        # 초기 변환 매개변수 설정 (회전과 이동 모두 고려)
        source_points = obs_points[-1]
        curr_indices = np.transpose(np.nonzero(curr)) - 512
        target_points = curr_indices
        obs_points.append(target_points)
        ##############################################################################

        ############################# odom pose #####################################
        #print(odom_traj[image_idx])
        #print(odom_traj[image_idx-1])
        last_odom_traj = odom_traj[image_idx+1-BUNDLE_NUM:image_idx+1] - odom_traj[image_idx+1-BUNDLE_NUM]
        last_odom_traj[:,1:] /=0.125
        dx = -last_odom_traj[:,2].copy()
        dy = -last_odom_traj[:,1].copy()
        last_odom_traj[:,1] = dx
        last_odom_traj[:,2] = dy
        print("part START_IDX", max(image_idx+1-BUNDLE_NUM, START_IDX), "image_idx", image_idx)

        ############################# lidar pose #####################################


        aprx_pose_est_start_time = time.time()

        last_matching = None
        for bundle_idx in range(BUNDLE_NUM -1, 0, -1):
            for jump_step in range(1,BUNDLE_NUM):
                source_img_idx = image_idx - bundle_idx
                target_img_idx = source_img_idx + jump_step
                source_idx = source_img_idx - image_idx + BUNDLE_NUM -1
                target_idx = target_img_idx - image_idx + BUNDLE_NUM -1
                if len(obs_points) < bundle_idx+1:
                    continue
                if target_img_idx > image_idx:
                    continue
                print()
                print(source_img_idx, target_img_idx)
                #print(source_idx, target_idx)

                tmp_source_points = obs_points[source_idx - BUNDLE_NUM]
                tmp_target_points = obs_points[target_idx - BUNDLE_NUM]


                diff_odom_traj = odom_traj[target_img_idx] - odom_traj[source_img_idx]

                tmp_odom_yaw = diff_odom_traj[0]
                tmp_odom_dx  = diff_odom_traj[2]/res
                tmp_odom_dy  = diff_odom_traj[1]/res

                tmp_odom_ds  = np.sqrt(tmp_odom_dx * tmp_odom_dx + tmp_odom_dy*tmp_odom_dy)
                tmp_odom_du  = -tmp_odom_ds * np.cos(tmp_odom_yaw)
                tmp_odom_dv  = -tmp_odom_ds * np.sin(tmp_odom_yaw)

                crsp_hash, crsp_set, rigid_transform = fe_matching(tmp_source_points, tmp_target_points,(tmp_odom_yaw, tmp_odom_du, tmp_odom_dv), one2one = True)
                crsp_hash_bundle.setdefault(source_idx, {})
                crsp_hash_bundle[source_idx][target_idx] = crsp_hash

                tmp_yaw = mat2yaw(rigid_transform[:2,:2])
                tmp_du =  rigid_transform[0,3]
                tmp_dv =  rigid_transform[1,3]


                if bundle_idx == 1 and jump_step == 1:
                    last_matching = (crsp_hash, crsp_set, rigid_transform)


        aprx_pose_est_end_time = time.time()

        crsp_hash, crsp_set, rigid_transform = last_matching

        # to homogenous coordinate
        hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
        transformed_source_points = np.dot(rigid_transform, hm_source_points).T

        icp_yaw = -mat2yaw(rigid_transform[:2,:2])
        icp_du =  rigid_transform[0,3]
        icp_dv =  rigid_transform[1,3]

        print("icp param yaw, du, dv", icp_yaw, icp_du, icp_dv)

        #print("########################################################################################### 2nd crsp_hash size", len(crsp_hash))

        crsp_source_points = source_points[crsp_set[:,0].ravel()]
        crsp_target_points = target_points[crsp_set[:,1].ravel()]

        ########################################################## relation view ####################################
        """
        delta_u = 512
        delta_v = 512
        # to homogenous coordinate
        hm_source_points = np.hstack((source_points[:], np.zeros((len(source_points),1)), np.ones((len(source_points),1)))).T
        transformed_source_points = np.dot(rigid_transform, hm_source_points).T

        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(curr)

        plt.subplot(1,2,2)
        plt.imshow(prev+curr)

        for crsp_source_point, crsp_target_point in zip(crsp_source_points, crsp_target_points):
            plt.plot([crsp_source_point[1]+delta_u, crsp_target_point[1]+delta_u], [crsp_source_point[0]+delta_v, crsp_target_point[0]+delta_v])
        
        plt.axis("equal")
        plt.show()
        """
        #############################################################################################################

        yaw = mat2yaw(rigid_transform[:2,:2])
        du =          rigid_transform[0,3]
        dv =          rigid_transform[1,3]


        image_heading -= yaw
        heading -= yaw
        R_mat = yaw2mat(image_heading)
        dt = np.dot(R_mat, np.array([du, dv]))
        print("du", du, "dv", dv, "yaw", yaw*180.0/np.pi, "heading", heading*180.0/np.pi)
        print("dx", dt[0], "dy", dt[1])
        # predicted curr lidar pose based 0th lidar frame
        obs_lidar_pose = [image_heading, obs_lidar_poses[-1][1] + dt[0], obs_lidar_poses[-1][2] + dt[1]]
        obs_lidar_poses.append(obs_lidar_pose)
        last_obs_lidar_poses = np.array(obs_lidar_poses[-BUNDLE_NUM:])
        last_obs_points = obs_points[-BUNDLE_NUM:]

        prev = curr
        match_std_li = []
        ############################## relation set ##################################
        if len(last_obs_images) < BUNDLE_NUM:
            continue

        relation_start_time = time.time()
        f2f_relations, i2f_relations = get_relations(last_obs_points, last_obs_lidar_poses, crsp_hash_bundle)
        relation_end_time = time.time()
        #print("relation duration ", relation_end_time - relation_start_time, "s")

        fe_enroll_start_time = time.time()
        color_index = 0
        points_2d = []
        mask = []
        for match_id in sorted(f2f_relations.keys()):
            match_cluster = []
            if len(f2f_relations[match_id].keys()) < MATCH_NUM:
                continue

            match_cluster = []
            
            for image_num in sorted(f2f_relations[match_id].keys()):
                fe_bundle_point = f2f_relations[match_id][image_num]
                match_cluster.append(fe_bundle_point)

            match_cluster = np.array(match_cluster)
            plt.plot(match_cluster[:,0], match_cluster[:,1])
            point_2d = np.mean(match_cluster, axis=0).ravel()
            point_2d = match_cluster[0].ravel()
            points_2d.append(point_2d)
            match_data.append(match_cluster)

        plt.show()

        fe_enroll_end_time = time.time()

        print("Relation count :", len(points_2d))

        ##############################################################################
        match_std_array = np.array(match_std_li)

        ############################# estimate   #####################################

        # vector space to affine space
        random_indices = np.random.choice(len(points_2d), size=300, replace=False)
        points_2d = np.array(points_2d).reshape(-1,2)
        selected_points_2d = points_2d[random_indices]

        # get observation(raw sensor data)
        observations = []
        for image_num in range(BUNDLE_NUM):
            observation = []
            
            for match_id in sorted(f2f_relations.keys()):
                if len(f2f_relations[match_id].keys()) < MATCH_NUM:
                    continue
                if image_num in f2f_relations[match_id].keys():
                    observation.append(i2f_relations[image_num][match_id].ravel()[:2])
                else:
                    observation.append([np.nan, np.nan])

            observation = np.array(observation)

            selected_observation = observation[random_indices]
            observations.append(selected_observation)

        # do bundle adjustment
        bundle_adjustment_start_time = time.time()
        opt_lidar_poses, opt_points_2d = bundle_adjustment(selected_points_2d, last_obs_lidar_poses, observations)
        bundle_adjustment_end_time = time.time()


        # get optimized lidar pose
        for image_num, (opt_lidar_pose, obs_lidar_pose) in enumerate(zip(opt_lidar_poses, last_obs_lidar_poses)):
            image_yaw = opt_lidar_pose[0]
            obs_yaw = obs_lidar_pose[0]
            image_position = opt_lidar_pose[1:]
            obs_position = obs_lidar_pose[1:]

            if image_num < BUNDLE_NUM-1:
                opt_lidar_locs.append([image_yaw, image_position[0], image_position[1]])
            if image_num == BUNDLE_NUM -1:
                opt_lidar_locs.append([image_yaw, image_position[0], image_position[1]])

        # draw optimized lidar trajectory
        opt_lidar_locs = np.array(opt_lidar_locs)

        if image_idx > START_IDX + BUNDLE_NUM-3:
            break


    obs_lidar_poses = np.array(obs_lidar_poses)

    image2real_res = 1
    real2image_res = 0.125
    start = 0

    # draw trajectory - odom
    plt.figure()
    plt.title("trajectory compare")
    plt.plot(    odom_traj[START_IDX:START_IDX+BUNDLE_NUM,1]/real2image_res, odom_traj[START_IDX:START_IDX+BUNDLE_NUM,2]/real2image_res, c="g", label ="odom")
    plt.scatter( odom_traj[START_IDX:START_IDX+BUNDLE_NUM,1]/real2image_res, odom_traj[START_IDX:START_IDX+BUNDLE_NUM,2]/real2image_res, c="g")
    
    # draw trajectory - observation
    plt.plot(    -obs_lidar_poses[-BUNDLE_NUM:,2]*image2real_res, -obs_lidar_poses[-BUNDLE_NUM:,1]*image2real_res,  c="r", label = "observation")
    plt.scatter( -obs_lidar_poses[-BUNDLE_NUM:,2]*image2real_res, -obs_lidar_poses[-BUNDLE_NUM:,1]*image2real_res,  c="r")

    # draw trajectory - bundle adjust
    plt.plot(   -opt_lidar_locs[:,2]*image2real_res, -opt_lidar_locs[:,1]*image2real_res, c = 'b', label = "opimize")
    plt.scatter(-opt_lidar_locs[:,2]*image2real_res, -opt_lidar_locs[:,1]*image2real_res, c = 'b')
    #print(odom_traj[:BUNDLE_NUM,0])

    #plt.scatter(points_2d[:,0], points_2d[:,1], s = 1)
    plt.legend()
    plt.axis("equal")

    # draw stiching image
    last_view_images = view_images[-BUNDLE_NUM:]
    result  = last_view_images[0].copy()
    result2 = last_view_images[0].copy()

    print("-----------------------------------------")
    headings = odom_traj[image_idx+1-BUNDLE_NUM:image_idx+1,0]
    last_odom_view = last_odom_traj.copy()
    last_odom_view[:,0] = headings
    print("last odom traj", last_odom_view)
    print("-----------------------------------------")
    print("last obs poses", last_obs_lidar_poses)
    print("-----------------------------------------")
    opt_lidar_locs[:] -= opt_lidar_locs[0]
    print("opt_lidar_locs", opt_lidar_locs)
    print("-----------------------------------------")
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

    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title("observation merge")
    plt.imshow(result)
    plt.axis("equal")

    plt.subplot(1,2,2)
    plt.title("bundle adjust merge")
    plt.imshow(result2)



    plt.axis("equal")

    plt.show()
    ###############################################################################

if __name__ == "__main__":
    main()
