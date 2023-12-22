import cv2
import numpy as np

# 이전 프레임과 현재 프레임 로드
previous_frame = cv2.imread('mat_alt_00100.png', cv2.IMREAD_GRAYSCALE)
current_frame = cv2.imread('mat_alt_00101.png', cv2.IMREAD_GRAYSCALE)

# ORB 디텍터 생성
orb = cv2.ORB_create()

# 이전 프레임에서 특징점 검출 및 기술자 계산
keypoints_previous, descriptors_previous = orb.detectAndCompute(previous_frame, None)

# 현재 프레임에서 특징점 검출 및 기술자 계산
keypoints_current, descriptors_current = orb.detectAndCompute(current_frame, None)

# 특징점 매칭
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors_previous, descriptors_current)

# 가장 좋은 매칭만 선택
matches = sorted(matches, key=lambda x: x.distance)

# 일부 매칭 결과만 선택 (가려진 지역의 특징점 재활용)
num_selected_matches = 150
good_matches = matches[:num_selected_matches]

# 선택한 매칭 결과를 사용하여 가려진 지역의 특징점 위치 업데이트
previous_points = np.float32([keypoints_previous[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
current_points = np.float32([keypoints_current[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

# Optical Flow 계산
flow = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, previous_points, None)

# Optical Flow 결과에서 새로운 특징점 위치 얻기
print(len(flow))
new_points = flow[0].reshape(-1, 2)

# 결과를 시각화
result_image = cv2.drawMatches(previous_frame, keypoints_previous, current_frame, keypoints_current, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 출력
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

