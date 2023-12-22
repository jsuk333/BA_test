import os
import sys
import numpy as np
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt

def tomasi(img):
    max_corners = 300
    quality_level = 0.01
    min_distance = 11
    corners = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)

    corner_pts = np.int0(corners)
    result_img = np.copy(img)
    for corner_pt in corner_pts:
        x, y = corner_pt.ravel()
        cv2.circle(result_img, (x,y), 3, 255,-1)

    return (corners, result_img)

def feature_matching(prev, curr):
    p0, detect_img = tomasi(prev)
    lk_params = dict( 
        winSize  = (30, 30),
        maxLevel = 5,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10000,
        0.003)
    ) # criteria type, maxCount, epsilon
    
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
    X = p0[status==1]
    y = p1[status==1]
    
    prev_pts = np.int0(X)
    curr_pts = np.int0(y)
    prev_img = np.copy(prev)
    curr_img = np.copy(curr)
    for prev_pt, curr_pt in zip(prev_pts, curr_pts):
        prev_x, prev_y = prev_pt.ravel()
        curr_x, curr_y = curr_pt.ravel()
    
        scale = (curr_x - prev_x)*(curr_x - prev_x) * (curr_y - prev_y)*(curr_y - prev_y)

        cv2.circle(prev_img, (prev_x, prev_y), 3, 255,-1)
        cv2.circle(curr_img, (curr_x, curr_y), 3, 255,-1)

    return (X, y, prev_img, curr_img)

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

def get_feature_points_set(prev_feature_points_set, curr_feature_points_set, R_mats, lidar_poses):
    prev_feature_set = []
    curr_feature_set = []
    for prev_feature_points, curr_feature_points, R_mat, lidar_pose in zip(prev_feature_points_set, curr_feature_points_set, R_mats, lidar_poses):
        tf_mat = np.array([
            [R_mat[0,0], R_mat[0,1], lidar_pose[0]],
            [R_mat[1,0], R_mat[1,1], lidar_pose[1]],
            [0.0,        0.0,        1.0],
        ])
        homogeneous_factor = np.ones(len(prev_feature_points)).reshape(-1,1)
        prev_feature_points = np.hstack((prev_feature_points, homogeneous_factor))
        curr_feature_points = np.hstack((curr_feature_points, homogeneous_factor))

        prev_feature_points = np.dot(tf_mat, prev_feature_points.T)
        curr_feature_points = np.dot(tf_mat, curr_feature_points.T)
        prev_feature_set.append(prev_feature_points[:2].T)
        curr_feature_set.append(curr_feature_points[:2].T)
    return prev_feature_set, curr_feature_set

def main():
    filenames = os.listdir()
    filenames = sorted(filenames)
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]

    init_img = cv2.imread(filenames[0])
    print(init_img.shape)
    prev = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, (5,5), 1)

    lidar_poses = [[0.0,0.0]]
    prev_feature_points_set = []
    curr_feature_points_set = []
    R_mats = []
    feature_points = []
    heading = 0.0

    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    end_flag = False
    bundle_num = 5
    colors = ["lightgrey", "silver", "darkgray", "gray", "dimgray"]
    for i, filename in enumerate(filenames[1:]):
        
        if "mat_alt_0" != filename[:9]:
            continue
        
        # image load
        new_img = cv2.imread(filename)
        curr = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        curr = cv2.GaussianBlur(curr, (5,5), 1)
        
        # feature_matching
        X, y, prev_img, curr_img = feature_matching(prev, curr)

        # estimation
        tvec, yaw = estimate(X,y)
        print("tvec", tvec, "yaw", yaw)

        # lidar pose
        heading -= yaw
        R = yaw2mat(heading)
        dt = np.dot(R, np.array([-tvec[1], -tvec[0]]))
        #dt = np.dot(R, np.array([0, 0]))

        lidar_pose = [lidar_poses[-1][0] + dt[0], lidar_poses[-1][1] + dt[1]]
        lidar_poses.append(lidar_pose)
        last_lidar_poses = np.array(lidar_poses[-bundle_num:])
        last_lidar_poses -= last_lidar_poses[0]

        R_mats.append(R)
        last_R_mats = R_mats[-bundle_num:]
        
        X -= np.array([512,512])
        y -= np.array([512,512])
        # feature point
        prev_feature_points_set.append(X)
        curr_feature_points_set.append(y)

        prev = curr
        
        last_prev_feature_points_set = prev_feature_points_set[-bundle_num:]
        last_curr_feature_points_set = curr_feature_points_set[-bundle_num:]

        prev_feature_set, curr_feature_set = get_feature_points_set(
                last_prev_feature_points_set, last_curr_feature_points_set, 
                last_R_mats, last_lidar_poses)

        #plt.figure()
        #plt.imshow(prev_img)
        #plt.figure()
        #plt.imshow(curr_img)


        plt.figure()
        color_index = 0
        # view
        for prev_pose, curr_pose, prev_features, curr_features in zip(last_lidar_poses[:-1], last_lidar_poses[1:], prev_feature_set, curr_feature_set ):
            print(color_index)
            for prev_feature, curr_feature in zip(prev_features, curr_features):
                line = np.vstack((prev_pose, prev_feature, curr_feature, curr_pose))
                plt.plot( line[:,0], line[:,1], c = colors[color_index])
            color_index += 1
            if color_index > 4:
                color_index = 0

        plt.plot(last_lidar_poses[:,0], last_lidar_poses[:,1], c = "r")
        plt.scatter(last_lidar_poses[:,0], last_lidar_poses[:,1], c = "r", s = 100)
        plt.show()

        # keboard event
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

        # prepare next



if __name__ == "__main__":
    main()
