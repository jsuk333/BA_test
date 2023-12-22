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

def orb(img):
    orb = cv2.ORB_create()
    keypoints, des = orb.detectAndCompute(img, None)
    for keypoint in keypoints:
        pt = (keypoint.pt).astype(np.uint8)
        cv2.circle(result_img, (x,y), 3, 255, -1)

    return keypoints, result_img

def feature_matching(prev, curr):
    p0, detect_img = tomasi(prev)
    lk_params = dict( winSize  = (20, 20),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.003))
    print(p0.shape)
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)
    X = p0[status==1]
    y = p1[status==1]
    
    prev_pts = np.int0(X)
    curr_pts = np.int0(y)
    valid_mask = []
    for prev_pt, curr_pt in zip(prev_pts, curr_pts):
        prev_x, prev_y = prev_pt.ravel()
        curr_x, curr_y = curr_pt.ravel()
    
        scale = (curr_x - prev_x)*(curr_x - prev_x) * (curr_y - prev_y)*(curr_y - prev_y)
        if scale < 400:
            valid_mask.append(True)
        else:
            valid_mask.append(True)
            
        #cv2.circle(detect_img, (prev_x, prev_y), 3, 255, -1)
        cv2.circle(detect_img, (curr_x, curr_y), 3, 255, -1)
        cv2.line(  detect_img, (prev_x, prev_y), (curr_x, curr_y), (255,255,255), 1)

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

    init_img = cv2.imread("mat_alt_00099.png")
    prev = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, (5,5), 1)
    hsv = np.zeros_like(init_img)
    hsv[..., 1] = 255

    trajectory = [[0.0,0.0]]
    heading = 0.0
    #cv2.namedWindow("image checker", flags=cv2.WINDOW_NORMAL)
    end_flag = False
    for i, filename in enumerate(filenames[108:]):
        
        if "mat_alt_0" != filename[:9]:
            continue
        print(filename)
        
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
        plt.imshow(result_img)
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


    cv2.destroyAllWindows()
    
    #trajectory = np.array(trajectory)
    #plt.plot(trajectory[:,0], trajectory[:,1])
    #plt.show()



if __name__ == "__main__":
    main()
