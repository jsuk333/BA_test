import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# 가상의 3D 포인트와 카메라 포즈 생성
num_points = 50
num_cameras = 5

# 가상의 3D 포인트 생성
points_3d = np.random.rand(num_points, 3) * 10

# 가상의 카메라 포즈 생성
camera_poses = []
for i in range(num_cameras):
    rotation = np.random.rand(3) * 360
    translation = np.random.rand(3) * 10
    pose = np.hstack([rotation.reshape(-1,1), translation.reshape(-1, 1)])
    print(pose)
    camera_poses.append(pose)

# 관측 데이터 생성
observations = []
for i in range(num_cameras):
    R_i = R.from_euler('xyz', camera_poses[i][:3,0], degrees=True).as_matrix()
    t_i = camera_poses[i][:3,1]
    proj_matrix_i = np.hstack([R_i, t_i.reshape(-1, 1)])
    proj_matrix_i = proj_matrix_i.T
    projections = np.dot(points_3d, proj_matrix_i.T)
    projections /= projections[:, 2].reshape(-1, 1)
    noise = np.random.normal(scale=0.1, size=(num_points, 2))
    noisy_projections = projections[:, :2] + noise
    print(noisy_projections)
    observations.append(noisy_projections)

# 번들 조정 함수 정의
def bundle_adjustment(points_3d, camera_poses, observations):
    def residual(params):
        # 카메라 포즈와 3D 포인트 위치 추출
        num_cameras = len(camera_poses)
        num_points = len(points_3d)
        camera_params = params[:num_cameras * 6].reshape(num_cameras, 6)
        point_params = params[num_cameras * 6:].reshape(num_points, 3)
        
        # 잔차 계산
        residuals = []
        for i in range(num_cameras):
            R_i = R.from_euler('xyz', camera_params[i, :3], degrees=True).as_matrix()
            t_i = camera_params[i, 3:]
            proj_matrix_i = np.hstack([R_i, t_i.reshape(-1, 1)])
            projected_points = np.dot(point_params, proj_matrix_i)
            projected_points /= projected_points[:, 2].reshape(-1, 1)
            residuals.append((observations[i] - projected_points[:, :2]).ravel())
        return np.concatenate(residuals)
    
    # 초기 매개변수 추정
    initial_params = np.hstack([pose.ravel() for pose in camera_poses] + [point_3d.ravel() for point_3d in points_3d])
    
    # 번들 조정 수행
    result = least_squares(residual, initial_params, verbose=2)
    
    # 결과 반환
    optimized_camera_poses = []
    for i in range(num_cameras):
        rotation = result.x[i * 6 : i * 6 + 3].reshape(-1, 1)
        translation = result.x[i * 6 + 3: i * 6 + 6]
        pose = np.hstack([rotation.reshape(-1,1), translation.reshape(-1, 1)])
        optimized_camera_poses.append(pose)
    
    optimized_points_3d = result.x[num_cameras * 6:].reshape(num_points, 3)
    
    return optimized_camera_poses, optimized_points_3d

# 번들 조정 수행
optimized_camera_poses, optimized_points_3d = bundle_adjustment(points_3d, camera_poses, observations)

# 결과 시각화
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(optimized_points_3d)
visualizer.add_geometry(pcd)
visualizer.run()
visualizer.destroy_window()

