import os
import sys
import numpy as np
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt

def farneback(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

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
    p1,_ = tomasi(curr)

    p0 = p0.reshape(-1, 2).astype(np.float32)
    p1 = p1.reshape(-1, 2).astype(np.float32)

    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=16, multi_probe_level=1)
    search_params = dict(checks =16)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(des1, des2)


    
    flann_params = dict(algorithm=6, trees=5)
    flannIndex = cv2.FlannBasedMatcher(flann_params, {})
    indices, distances = flannIndex.knnSearch(p0, 2, params={})
    
    X = []
    y = []

    for i in range(indices.shape[0]):
        for j in range(2):
            index = indices[i, j]
            distance = np.sqrt((prev_points[i, 0] - curr_points[index, 0]) ** 2 + (prev_points[i, 1] - curr_points[index, 1]) ** 2)

            if distance < 20.0:
                X.append((prev_points[i, 0], prev_points[i, 1]))
                y.append((curr_points[index, 0], curr_points[index, 1]))

    
    prev_pts = np.int0(X)
    curr_pts = np.int0(y)
    valid_mask = []
    for prev_pt, curr_pt in zip(prev_pts, curr_pts):
        prev_x, prev_y = prev_pt.ravel()
        curr_x, curr_y = curr_pt.ravel()
    
        scale = (curr_x - prev_x)*(curr_x - prev_x) * (curr_y - prev_y)*(curr_y - prev_y)
        if scale < 200:
            valid_mask.append(True)
        else:
            valid_mask.append(True)
            continue

        #cv2.circle(detect_img, (prev_x, prev_y), 3, 255,-1)
        cv2.circle(detect_img, (curr_x, curr_y), 3, 255,-1)
        cv2.line(  detect_img, (prev_x, prev_y), (curr_x, curr_y), (255,255,255), 2)

    valid_mask = np.array(valid_mask)
    X = X[valid_mask]
    y = y[valid_mask]
    result_img = detect_img
    return (X, y, result_img)


def estimate(X, y):
    
    coef, inlier_mask = cv2.estimateAffine2D(X, y)

    tvec = coef[:,2]
    rotmat = coef[:2,:2]
    
    cs = rotmat[0,0]
    ss = rotmat[1,0]
    yaw = np.arctan2(ss, cs)
    return (tvec, yaw)



def main():
    filenames = os.listdir()
    filenames = sorted(filenames)

    init_img = cv2.imread("mat_alt_00100.png")
    prev = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, (5,5), 1)
    hsv = np.zeros_like(init_img)
    hsv[..., 1] = 255

    trajectory = [[0.0,0.0]]
    heading = 0.0
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    for i, filename in enumerate(filenames[100:]):
        
        if "mat_alt_0" != filename[:9]:
            continue
        #print(filename)
        
        # image load
        new_img = cv2.imread(filename)
        curr = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        curr = cv2.GaussianBlur(curr, (7,7), 3)
        
        # feature_matching
        X, y, matching_img = feature_matching(prev, curr)

        #estimation
        tvec, yaw = estimate(X,y)
        #print("tvec", tvec, "yaw", yaw)
        heading -= yaw
        R = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
                ])
        dt = np.dot(R, np.array([-tvec[1], -tvec[0]]))
        point = [trajectory[-1][0] + dt[0], trajectory[-1][1] + dt[1]]
        trajectory.append(point)
        if i% 100 == 0:
            print(point)

        # etc
        prev = curr

        result_img = matching_img
        cv2.imshow("image checker", result_img)
        key_code = cv2.waitKey(0) & 0XFF

        if key_code in (ord(']'), ord('d')): # right (->)
            print("next")
        elif key_code in (27, ord('q')):
            print("Exit")
            break


    cv2.destroyAllWindows()
    
    #trajectory = np.array(trajectory)
    #plt.plot(trajectory[:,0], trajectory[:,1])
    #plt.show()



if __name__ == "__main__":
    main()
