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

BLUR_SIZE = (5,5)
BLUR_VAR = 5

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
            reverse_R_i = yaw2mat(-lidar_params[i, 0])
            reverse_t_i = -lidar_params[i, 1:]

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
            residual = np.sum((observations[i+1] - projected_points[:, :2]).ravel())
            residuals.append((observations[i+1] - projected_points[:, :2]).ravel())
        return np.concatenate(residuals)
    
    num_lidars = len(lidar_poses) -1

    # 초기 매개변수 추정
    #lidar_poses -= lidar_poses[0]
    initial_params = np.hstack([pose.ravel() for pose in lidar_poses[1:]])
    
    # 번들 조정 수행
    result = least_squares(residual, initial_params, method = "lm", verbose=2)

    # 결과 반환
    optimized_lidar_poses = [lidar_poses[0].ravel()]
    for i in range(num_lidars):
        pose = result.x[i * 3:i*3 + 3]
        optimized_lidar_poses.append(pose)


    return optimized_lidar_poses

def extract_feature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, BLUR_SIZE, BLUR_VAR)

    canny_threshold_low =50
    canny_threshold_high = 110
    edge = cv2.Canny(img, canny_threshold_low, canny_threshold_high)
    
    img += edge
    img = cv2.GaussianBlur(edge, BLUR_SIZE, BLUR_VAR)

    corners = cv2.goodFeaturesToTrack(img, maxCorners=1000, qualityLevel=0.001, minDistance=11, blockSize =30, useHarrisDetector = False)
    #print("num_nonzero", np.nonzero(edge)[0].shape)
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #nonzero_indices = np.transpose(np.nonzero(grey))
    #corners = nonzero_indices.reshape(-1,1,2)


    return corners

def get_descriptor(img, corners):

    # ORB 객체 생성
    orb = cv2.ORB_create(**ORB_PARAMS)
    
    # 모서리 포인트를 특징 포인트로 변환
    keypoints = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size = 10) for c in corners]
    
    # ORB 디스크립터 추출
    keyPoints, des = orb.compute(img, keypoints)

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

def mat2yaw(R):
    yaw = np.arctan2(-R[1,0], R[0,0])
    return yaw

# find relation between sensor, feature
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
                if scale > 100:
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
            
            # based first image frame
            f2f_relations.setdefault(match_idx, {})
            f2f_relations[match_idx][image_num] = fe_bundle_point
            
            # based selected image frame
            i2f_relations.setdefault(image_num, {})
            i2f_relations[image_num][match_idx] = fe_raw_point[:2]

    return f2f_relations, i2f_relations

def load_trajectory(filename):

    trajectory = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(" ")
            timestamp = float(tokens[0])
            yaw = float(tokens[1])
            x = float(tokens[2])
            y = float(tokens[3])
            z = float(tokens[4])
            trajectory.append([yaw, x, y])

    return np.array(trajectory)


def main():
    filenames = os.listdir()
    filenames = sorted(filenames)
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]

    odom_traj = load_trajectory("odom_traj.txt")
    odom_traj -= odom_traj[0]

    start_idx = 90
    print(filenames[start_idx])
    prev = cv2.imread(filenames[start_idx])
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
    prev_view_image = prev.copy()
    view_images = []
    heading = 0.0

    # viewer setting
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    end_flag = False
    colors = ["gainsboro", "silver", "darkgrey", "grey", "dimgrey"]
    
    # bundel adjustment set
    bundle_num = 5
    keyPoint_table = []
    f2i_idx_relations = {}

    # extract initial feature
    corners = extract_feature(prev)
    kep, des = get_descriptor(prev, corners)
    keyPoint_table.append((kep, des))

    for corner in corners:
        u, v = corner.ravel()
        cv2.circle(prev_view_image, (int(u),int(v)), 2, (255,0,0), -1)

    view_images.append(prev_view_image)

    # main loop
    opt_lidar_locs = []
    for image_idx, filename in enumerate(filenames[start_idx+1:], start = start_idx+1):

        ############################# image load #####################################
        # image load
        curr = cv2.imread(filename)
        images.append(curr.copy())
        last_obs_images = images[-bundle_num:]
        ##############################################################################

        ############################# fe_matching ####################################

        # 1st feature extract
        kep1, des1 = keyPoint_table[-1]
        corners = extract_feature(curr)
        kep2, des2 = get_descriptor(curr, corners)

        # 1st feature matching
        matches = fe_matching(des1, des2)

        X = np.array([kep1[match.queryIdx].pt for match in matches])
        y = np.array([kep2[match.trainIdx].pt for match in matches])

        # missed feature tracking
        all_query_indices = set(range(len(kep1)))
        matched_query_indices = set(match.queryIdx for match in matches)
        not_matched_indices = list(all_query_indices - matched_query_indices)

        p0 = np.float32([kep1[not_matched_indice].pt for not_matched_indice in not_matched_indices]).reshape(-1,1,2)

        lk_params = dict( winSize  = (40, 40),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.003))

        p0.astype(np.float32)

        p1, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
        X_o = p0[status==1]
        y_o = p1[status==1]

        X = np.vstack((X, X_o))
        y = np.vstack((y, y_o))

        corners = np.vstack((corners, y_o.reshape(-1,1,2)))
        kep2, des2 = get_descriptor(curr, corners)
        keyPoint_table.append((kep2, des2))
        view_image = curr.copy()

        for corner in corners:
            u, v = corner.ravel()
            cv2.circle(view_image, (int(u),int(v)), 2, (0,0,(image_idx-start_idx)*50%255), -1)

        view_images.append(view_image)
        ##############################################################################

        ############################# estimation #####################################
        tvec, yaw = estimate(X,y)
        ##############################################################################

        ############################# lidar pose #####################################
        heading += yaw
        print("tvec", tvec*0.125, "yaw", yaw*180.0/np.pi, "heading", heading*180.0/np.pi)
        R_mat = yaw2mat(heading)
        dt = np.dot(R_mat, np.array([tvec[0], tvec[1]]))
        
        # predicted curr lidar pose based 0th lidar frame
        obs_lidar_pose = [heading, obs_lidar_poses[-1][1] + dt[0], obs_lidar_poses[-1][2] + dt[1]]
        obs_lidar_poses.append(obs_lidar_pose)
        last_obs_lidar_poses = np.array(obs_lidar_poses[-bundle_num:])
        #local_obs_lidar_poses = last_obs_lidar_poses - last_obs_lidar_poses[-2]
        #last_obs_lidar_poses[:,1:] -= last_obs_lidar_poses[0,1:]

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
        color_index = 0
        points_2d = []
        for match_id in sorted(f2f_relations.keys()):
            match_cluster = []
            if len(f2f_relations[match_id].keys()) < bundle_num:
                print(f2f_relations[match_id])
                continue

            for image_num in f2f_relations[match_id].keys():
                fe_point = f2f_relations[match_id][image_num]
                match_cluster.append(fe_point)

            match_cluster = np.array(match_cluster)
            pd_point = np.mean(match_cluster, axis = 0)
            #print("match mean", pd_point, "match_std", np.std(match_cluster, axis=0))
            #match_std = np.sum(np.std(match_cluster, axis = 0))
            #if match_std > 5:
            #    del f2f_relations[match_id]
            #    continue
            points_2d.append(pd_point)
    
        if len(points_2d) < 10:
            print("points_2d", len(points_2d))
            continue

        ##############################################################################

        if len(last_obs_images) < bundle_num:
            continue

        ############################# estimate   #####################################

        # vector space to affine space
        points_2d = np.array(points_2d).reshape(-1,2)

        # get observation(raw sensor data)
        observations = []
        for image_num in range(bundle_num):
            observation = []
            
            for match_id in sorted(f2f_relations.keys()):
                if len(f2f_relations[match_id].keys()) < bundle_num:
                    continue
                observation.append(i2f_relations[image_num][match_id].ravel()[:2])

            observation = np.array(observation)
            observations.append(observation)

        # do bundle adjustment
        opt_lidar_poses = bundle_adjustment(points_2d, last_obs_lidar_poses, observations)

        opt_lidar_poses -= opt_lidar_poses[-2]

        # get optimized lidar pose
        last_opt_lidar_pose = [0,0,0]
        for image_num, (opt_lidar_pose, obs_lidar_pose) in enumerate(zip(opt_lidar_poses, obs_lidar_poses)):
            image_yaw = opt_lidar_pose[0]
            obs_yaw = obs_lidar_pose[0]
            print("image_num", image_num, "image_yaw", image_yaw)
            image_position = opt_lidar_pose[1:]
            obs_position = obs_lidar_pose[1:]
            #print("bundle_tvec", image_pose*0.125, "bundle_yaw", image_yaw*180.0/np.pi)

            #for fe_point in points_2d:
            #    plt.plot([obs_position[0], fe_point[0]], [obs_position[1], fe_point[1]], c = colors[image_num])

            #for fe_point in opt_points_2d:
            #    plt.plot([image_position[0], fe_point[0]], [image_position[1], fe_point[1]], c = colors[image_num])

            print("image_num", image_num)
            if len(opt_lidar_locs) == 0:
                opt_lidar_locs.append([0, 0, 0])
            elif len(opt_lidar_locs) < bundle_num:
                opt_lidar_poses -= opt_lidar_poses[0]
                opt_lidar_pose = opt_lidar_poses[image_num]
                opt_lidar_locs.append(opt_lidar_pose)
            elif len(opt_lidar_locs) >= bundle_num:
                
                opt_lidar_locs.append([image_yaw, image_position[0], image_position[1]])

            last_opt_lidar_pose = opt_lidar_locs[-1]

        # draw optimized lidar trajectory
        print("index", image_idx)
        trajectory = np.array(opt_lidar_locs)
        #plt.plot(trajectory[:,0], trajectory[:,1], "r")

        print("image_idx", image_idx, "bundle_num", bundle_num)
        if image_idx > bundle_num-2:
            break

        ##############################################################################

        ############################# keyboard event #################################
#        while True:
#            command = input("next?")
#
#            if command in ["o", "n"]:
#                break
#            elif command in ["x", "q"]:
#                end_flag = True
#                print("System end!")
#                break
#            else:
#                pass
#
#        if end_flag:
#            break

    obs_lidar_poses = np.array(obs_lidar_poses)
    
    yut = len(obs_lidar_poses)
    res = 0.125

    plt.plot(odom_traj[:yut,1]/res, odom_traj[:yut,2]/res, "g")
    plt.scatter(odom_traj[:yut,1]/res, odom_traj[:yut,2]/res)
    #plt.plot(obs_lidar_poses[:,1], obs_lidar_poses[:,2], 'b')
    plt.plot(trajectory[:,1], trajectory[:,2], 'b')

    result = np.zeros_like(view_images[0])

    a = np.zeros_like(view_images[0])
    for i, image in enumerate(view_images):
        theta = -odom_traj[i,0]
        tx = odom_traj[i,1]/0.125
        ty = odom_traj[i,2]/0.125

        # 이미지의 중심을 계산
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전을 위한 변환 행렬 생성
        rotation_angle = theta  # 회전 각도
        scale_factor = 1.0   # 스케일 인자
        
        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
        print("rotation",rotation_matrix)

        # 회전 및 이동 적용
        rotation_matrix[0,2] += tx
        rotation_matrix[1,2] += ty
        warp1 = cv2.warpAffine(image, rotation_matrix, (w, h))

        #a += warp1
        a += image

        print("view image num", i)
        theta = trajectory[i,0]/np.pi*180
        tx = trajectory[i,1]
        ty = trajectory[i,2]

        # 이미지의 중심을 계산
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전을 위한 변환 행렬 생성
        rotation_angle = theta  # 회전 각도
        scale_factor = 1.0   # 스케일 인자

        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, scale_factor)
        print("rotation",rotation_matrix)
        rotated_image2 = cv2.warpAffine(image, rotation_matrix, (w, h))


        translation_matrix = np.array([[1, 0, -ty], [0, 1, -tx]])
        warp2 = cv2.warpAffine(rotated_image2, translation_matrix, (w, h))
        result += warp2.astype(np.uint8)

    plt.figure()
    plt.imshow(a)

    plt.figure()
    plt.imshow(result)

    plt.axis("equal")
    plt.show()

        ###############################################################################

if __name__ == "__main__":
    main()
