import os
import time
import sys
import numpy as np
import cv2
from sklearn import linear_model
#import matplotlib.pyplot as plt

ORB_PARAMS = dict(
            nfeatures = 2000,
            scaleFactor = 3,
            nlevels = 3,
            edgeThreshold = 10,
            firstLevel = 0,
            WTA_K = 4,
            scoreType = cv2.ORB_HARRIS_SCORE,
            #scoreType = cv2.ORB_FAST_SCORE,
            patchSize = 31,
            fastThreshold = 5
        )

BLUR_SIZE = (5,5)
BLUR_VAR = 5

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


def orb(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, BLUR_SIZE, BLUR_VAR)


    canny_threshold_low =50
    canny_threshold_high = 110
    edge = cv2.Canny(img, canny_threshold_low, canny_threshold_high)

    #img = cv2.GaussianBlur(edge, BLUR_SIZE, BLUR_VAR)
    img = cv2.GaussianBlur(img, BLUR_SIZE, BLUR_VAR)

    corners = cv2.goodFeaturesToTrack(img, maxCorners=2000, qualityLevel=0.001, minDistance=11, blockSize =21, useHarrisDetector = False)
    #corners = np.int0(corners)

    # ORB 객체 생성
    orb = cv2.ORB_create(**ORB_PARAMS)
    
    # 모서리 포인트를 특징 포인트로 변환
    keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size = 10) for c in corners]
    
    # ORB 디스크립터 추출
    keyPoints, des = orb.compute(img, keypoints)

    # orb = cv2.ORB_create(**ORB_PARAMS)
    # keyPoints, des = orb.detectAndCompute(edge, None)

    corners = [[list(keyPoint.pt)] for keyPoint in keyPoints]
    corners = np.array(corners, dtype= np.float32)

    corner_pts = np.int0(corners)
    result_img = np.copy(img)
    for corner_pt in corner_pts:
        x, y = corner_pt.ravel()
        cv2.circle(result_img, (x,y), 3, 255,-1)

    return (corners, keyPoints, des, img)

def feature_matching(prev, curr):
    p0, kep1, des1, prev = orb(prev)
    p1, kep2, des2, curr = orb(curr)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=16, multi_probe_level=1)
    search_params = dict(checks =16)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(des1, des2)

    all_query_indices = set(range(len(kep1)))
    matched_query_indices = set(match.queryIdx for match in matches)
    not_matched_indices = list(all_query_indices - matched_query_indices)

    p0 = np.float32([kep1[not_matched_indice].pt for not_matched_indice in not_matched_indices]).reshape(-1,1,2)

    lk_params = dict( winSize  = (20, 20),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.003))
    p0.astype(np.float32)
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
    X_o = p0[status==1]
    y_o = p1[status==1]


    X   = np.array([kep1[match.queryIdx].pt for match in matches])
    y   = np.array([kep2[match.trainIdx].pt for match in matches])
    print(len(X_o))

    #X = np.vstack((X, X_o))
    #y = np.vstack((y, y_o))



    prev_pts = np.int0(X)
    curr_pts = np.int0(y)
    valid_mask = []
    prev_img = np.copy(prev)
    curr_img = np.copy(curr)
    max_scale = 0
    i = 0
    for prev_pt, curr_pt in zip(prev_pts, curr_pts):
        prev_x, prev_y = prev_pt.ravel()
        curr_x, curr_y = curr_pt.ravel()
    
        scale = (curr_x - prev_x)*(curr_x - prev_x) + (curr_y - prev_y)*(curr_y - prev_y)
        if scale>4000:
            continue

        if scale >max_scale:
            max_scale = scale
            #print("max_scale", max_scale, prev_pt, curr_pt)

        cv2.circle(prev_img, (prev_x, prev_y), 3, 255,-1)
        cv2.circle(curr_img, (curr_x, curr_y), 3, 255,-1)
        
        #cv2.line(  prev_img, (prev_x, prev_y), (curr_x, curr_y), (255,255,255), 2)
        i += 1
    return (X, y, prev_img, curr_img)


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
    filenames = [filename  for filename in filenames if filename[:9] == "mat_alt_0"]
    prev = cv2.imread(filenames[99])
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255

    trajectory = [[0.0,0.0]]
    heading = 0.0
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    index = 100
    while True:
        filename = filenames[index]
        match_start = time.time()   
        print(filename, index)
        
        # image load
        curr = cv2.imread(filename)

        # feature_matching
        X, y, prev_img, curr_img = feature_matching(prev, curr)
        
        #estimation
        tvec, yaw = estimate(X,y)
        heading -= yaw
        print("tvec", tvec, "yaw", yaw/np.pi*180.0, "heading", heading/np.pi*180.0)
        R = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        dt = np.dot(R, np.array([-tvec[1], -tvec[0]]))
        point = [trajectory[-1][0] + dt[0], trajectory[-1][1] + dt[1]]
        trajectory.append(point)
        if index% 100 == 0:
            print(point)

        match_end = time.time()
        print("match_time", match_end - match_start)
        # etc
        prev = curr

        cv2.imshow("prev_img", prev_img)
        cv2.imshow("curr_img", curr_img)
        key_code = cv2.waitKey(0) & 0XFF

        if key_code in (ord(']'), ord('d')): # right (->)
            print("next")
            index += 1

        if key_code in (ord('['), ord('u')): # left (<-)
            print("prev")
            index -= 1

        elif key_code in (27, ord('q')):
            print("Exit")
            break
        else:
            continue

    cv2.destroyAllWindows()
    



if __name__ == "__main__":
    main()
