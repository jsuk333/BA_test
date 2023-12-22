import os
import sys
import numpy as np
import cv2
from sklearn import linear_model
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

ORB_PARAMS = dict(
            nfeatures = 1000,
            scaleFactor = 2,
            nlevels = 3,
            edgeThreshold = 7,
            firstLevel = 0,
            WTA_K = 4,
            #scoreType = cv2.ORB_HARRIS_SCORE,
            scoreType = cv2.ORB_FAST_SCORE,
            patchSize = 30,
            fastThreshold = 5
        )


LK_PARAMS = dict( 
    winSize  = (50, 50),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10000,
    0.003)
) # criteria type, maxCount, epsilon

BLUR_SIZE = (5,5)
BLUR_VAR = 3

class BAData:
    def __init__(self):
        self.lidar_poses = []  # 카메라 포즈 저장
        self.feature_points = []  # 3D 특징점 좌표 저장
        self.matches = []  # 포즈와 특징점 간의 매칭 포인트 저장

    def add_lidar_pose(self, pose):
        # pose는 카메라의 위치와 방향 정보를 담고 있는 데이터로 예를 들면 [x, y, z, yaw, pitch, roll] 형식일 수 있습니다.
        self.lidar_poses.append(pose)

    def add_feature_point(self, point):
        # point는 3D 특징점의 좌표를 담고 있는 데이터로 예를 들면 [x, y, z] 형식일 수 있습니다.
        self.feature_points.append(point)

    def add_match(self, lidar_idx, point_idx, image_point):
        # lidar_idx는 lidar 인덱스, point_idx는 특징점 인덱스, image_point는 이미지에서의 좌표입니다.
        self.matches.append((lidar_idx, point_idx, image_point))

# 번들 조정 함수 정의
def bundle_adjustment(points_2d, lidar_poses, observations):

    def residual(params):
        # 카메라 포즈와 3D 포인트 위치 추출
        num_lidars = len(lidar_poses)
        num_points = len(points_2d)
        lidar_params = params[:num_lidars * 3].reshape(num_lidars, 3)
        point_params = params[num_lidars * 3:].reshape(num_points, 2)
        
        # 잔차 계산
        residuals = []
        for i in range(num_lidars):
            reverse_R_i = np.linalg.inv(yaw2mat(lidar_params[i, 0]))
            reverse_t_i = -lidar_params[i, 1:]
            proj_matrix_i = np.hstack([reverse_R_i, reverse_t_i.reshape(-1, 1)])
            projected_points = np.dot(point_params, proj_matrix_i)
            residuals.append((observations[i] - projected_points[:, :2]).ravel())
        return np.concatenate(residuals)
    
    num_lidars = len(lidar_poses)
    num_points = len(points_2d)
    # 초기 매개변수 추정
    initial_params = np.hstack([pose.ravel() for pose in lidar_poses] + [point_2d.ravel() for point_2d in points_2d])
    
    # 번들 조정 수행
    result = least_squares(residual, initial_params, verbose=2)
    
    # 결과 반환
    optimized_lidar_poses = []
    for i in range(num_lidars):
        rotation = yaw2mat(result.x[i * 3])
        translation = result.x[i * 3 + 1: i * 3 + 3].reshape(1,2)
        pose = np.vstack((rotation, translation))
        optimized_lidar_poses.append(pose)

    optimized_points_2d = result.x[num_lidars * 3:].reshape(num_points, 2)
    
    return optimized_lidar_poses, optimized_points_2d

def orb(img):
    orb = cv2.ORB_create(**ORB_PARAMS)
    keyPoints, des = orb.detectAndCompute(img, None)

    return (keyPoints, des)

def fe_matching(des1, des2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=16, multi_probe_level=1)
    search_params = dict(checks =16)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)

    #matches = flann.knnMatch(des1, des2, k=2)
    #filtered_matches = [m for (m, n) in matches]
    #return (kep1, kep2, matches, filtered_matches)

    return matches

def estimate(X, y):
    
    coef, inlier_mask = cv2.estimateAffine2D(X, y)

    tvec = coef[:,2]
    rotmat = coef[:2,:2]
    
    cs = rotmat[0,0]
    ss = rotmat[1,0]
    yaw = np.arctan2(ss, cs)
    return (tvec, yaw)

def yaw2mat(angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
            ])
    return R

def get_relations(keyPoint_table, obs_tf_mats):
    match_idx = 0
    f2i_idx_relations = {}
    match_count = 0
    f2f_relations = {}
    i2f_relations = {}


    for i in range(len(keyPoint_table)):
        kep1, des1 = keyPoint_table[i]
        for j in range(i+1, len(keyPoint_table)):
            kep2, des2 = keyPoint_table[j]
            matches = fe_matching(des1, des2)
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx

                query_x, query_y = kep1[query_idx].pt
                train_x, train_y = kep2[train_idx].pt

                scale = (train_x - query_x)*(train_x - query_x) + (train_y - query_y)*(train_y - query_y)
                if scale > 2000:
                     continue

                new_match_flag = True

                for tmp_match_idx in f2i_idx_relations.keys():
                    
                    if i in f2i_idx_relations[tmp_match_idx].keys():
                        if query_idx == f2i_idx_relations[tmp_match_idx][i]:
                            match_idx = tmp_match_idx
                            new_match_flag = False

                    if j in f2i_idx_relations[tmp_match_idx].keys():
                        if train_idx == f2i_idx_relations[tmp_match_idx][j]:
                            match_idx = tmp_match_idx
                            new_match_flag = False

                if new_match_flag:
                    f2i_idx_relations.setdefault(match_count, {})
                    match_count += 1

                f2i_idx_relations[match_idx][i] = query_idx
                f2i_idx_relations[match_idx][j] = train_idx

                match_idx = match_count

    fe_points = {}
    for match_idx in f2i_idx_relations.keys():
        for image_num in f2i_idx_relations[match_idx].keys():
            kep, _ = keyPoint_table[image_num]
            
            tf_mat = obs_tf_mats[image_num]
            key_idx = f2i_idx_relations[match_idx][image_num]
            x, y = kep[key_idx].pt
            fe_raw_point = np.array([x - 512, y - 512, 1]).reshape(-1, 3)
            fe_bundle_point = np.dot(tf_mat, fe_raw_point.T)
            fe_bundle_point = fe_bundle_point[:2]
            
            f2f_relations.setdefault(match_idx, {})
            f2f_relations[match_idx][image_num] = fe_bundle_point
            
            i2f_relations.setdefault(image_num, {})
            i2f_relations[image_num][match_idx] = fe_bundle_point

    return f2f_relations, i2f_relations

def main():
    filenames = os.listdir()
    filenames = sorted(filenames)
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]

    init_img = cv2.imread(filenames[0])
    prev = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, BLUR_SIZE, BLUR_VAR)
    obs_lidar_poses = [[0.0, 0.0,0.0]]
    global_lidar_poses = [[0.0, 0.0,0.0]]

    prev_obs_fe_points_set = []
    curr_obs_fe_points_set = []
    obs_rot_mats = []
    obs_tf_mats = [np.array([
        [1.0,   0.0,    0],
        [0.0,   1.0,    0],
        [0.0,   0.0,    1.0]
    ])]

    obs_fe_points = []
    images = [prev]
    heading = 0.0

    # viewer setting
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    end_flag = False
    colors = ["gainsboro", "silver", "darkgrey", "grey", "dimgrey"]
    
    # bundel adjustment set
    bundle_num = 5
    keyPoint_table = []
    f2i_idx_relations = {}

    kep, des = orb(prev)
    keyPoint_table.append((kep, des))

    for i, filename in enumerate(filenames[1:]):

        ############################# image load #####################################
        # image load
        new_img = cv2.imread(filename)
        curr = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        curr = cv2.GaussianBlur(curr, BLUR_SIZE, BLUR_VAR)
        images.append(curr)
        last_obs_images = images[-bundle_num:]
        ##############################################################################

        ############################# fe_matching ####################################
        kep1, des1 = keyPoint_table[-1]
        kep2, des2 = orb(curr)
        matches = fe_matching(des1, des2)
        keyPoint_table.append((kep2, des2))

        X = np.array([kep1[match.queryIdx].pt for match in matches])
        y = np.array([kep2[match.trainIdx].pt for match in matches])
        ##############################################################################

        ############################# estimation #####################################
        tvec, yaw = estimate(X,y)
        print("tvec", tvec, "yaw", yaw)
        ##############################################################################

        ############################# lidar pose #####################################
        heading -= yaw
        R_mat = yaw2mat(heading)
        dt = np.dot(R_mat, np.array([-tvec[1], -tvec[0]]))

        obs_lidar_pose = [heading, obs_lidar_poses[-1][0] + dt[0], obs_lidar_poses[-1][1] + dt[1]]
        obs_lidar_poses.append(obs_lidar_pose)
        last_obs_lidar_poses = np.array(obs_lidar_poses[-bundle_num:])
        last_obs_lidar_poses[:,1:] -= last_obs_lidar_poses[0,1:]
        tf_mat = np.array([
            [R_mat[0,0], R_mat[0,1], obs_lidar_pose[0]],
            [R_mat[1,0], R_mat[1,1], obs_lidar_pose[1]],
            [0.0,        0.0,        1.0],
        ])

        obs_rot_mats.append(R_mat)
        obs_tf_mats.append(tf_mat)
        last_obs_rot_mats = obs_rot_mats[-bundle_num:]
        last_obs_tf_mats = obs_tf_mats[-bundle_num:]
        last_keyPoint_table = keyPoint_table[-bundle_num:]
        prev = curr
        ##############################################################################
        
        ############################## relation set ##################################
        f2f_relations, i2f_relations = get_relations(last_keyPoint_table, last_obs_tf_mats)
        plt.figure()
        color_index = 0
        points_2d = []
        for match_id in f2f_relations.keys():
            match_cluster = []
            if len(f2f_relations[match_id].keys()) < len(last_keyPoint_table):
                continue
            for image_num in f2f_relations[match_id].keys():
                fe_point = f2f_relations[match_id][image_num]
                match_cluster.append(fe_point)

            match_cluster = np.array(match_cluster)
            pd_point = np.mean(match_cluster, axis = 0)
            points_2d.append(pd_point)
            #plt.plot(match_cluster[:,0], match_cluster[:,1])

        #for image_num in i2f_relations.keys():
        #    match_spoke = []
        #    image_pose = last_obs_lidar_poses[image_num]
        #    for match_id in i2f_relations[image_num].keys():
        #        fe_point = i2f_relations[image_num][match_id]
        #        plt.plot([image_pose[1], fe_point[0]], [image_pose[2], fe_point[1]], c = colors[image_num])

        #plt.axis("equal")
        #plt.show()

        if len(points_2d) < 10:
            continue

        ##############################################################################

        if len(last_obs_images) < bundle_num:
            continue

        ############################# estimate   #####################################
        points_2d = np.array(points_2d).reshape(-1,2)
        side = np.ones((len(points_2d ), 1))
        hmg_points_2d = np.hstack((points_2d, side))

        observations = []
        for i in range(len(last_obs_lidar_poses)):
            R_i = yaw2mat(last_obs_lidar_poses[i, 0])
            reverse_t_i = -last_obs_lidar_poses[i, 1:].reshape(1,-1)
            proj_matrix_i = np.vstack((R_i, reverse_t_i))
            projected_points = np.dot(hmg_points_2d, proj_matrix_i)
            observations.append(projected_points[:,:2])

        opt_lidar_poses, opt_points_2d = bundle_adjustment(points_2d, last_obs_lidar_poses, observations)
        opt_lidar_locs = []
        for image_num, opt_lidar_pose in enumerate(opt_lidar_poses):
            image_pose = opt_lidar_pose[2,:]
            for fe_point in opt_points_2d:
                opt_lidar_locs.append([image_pose[0], image_pose[1]])
                plt.plot([image_pose[0], fe_point[0]], [image_pose[1], fe_point[1]], c = colors[image_num])
        
        opt_lidar_locs = np.array(opt_lidar_locs)
        plt.plot(opt_lidar_locs[:,0], opt_lidar_locs[:,1], c = "r")

        plt.axis("equal")
        plt.show()

        ##############################################################################

        ############################# keyboard event #################################
        while True:
            command = input("next?")

            if command in ["o", "n"]:
                break
            elif command in ["x", "q"]:
                end_flag = True
                print("System end!")
                break
            else:
                pass

        if end_flag:
            break

        ###############################################################################

if __name__ == "__main__":
    main()
