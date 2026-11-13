import pandas as pd
import numpy as np
import open3d as o3d

#파일 읽기
chair = pd.read_csv("Chair.csv", skiprows=[1])
board = pd.read_csv("Board.csv", skiprows=[1])

chair_points = chair[["X", "Y", "Z"]].to_numpy()
board_points = board[["X", "Y", "Z"]].to_numpy()

chair_centroid = np.mean(chair_points, axis=0)
board_centroid = np.mean(board_points, axis=0)

diff = chair_centroid - board_centroid
distance = np.linalg.norm(diff)

merged_points = np.vstack((chair_points, board_points))

chair_pcd = o3d.geometry.PointCloud()
chair_pcd.points = o3d.utility.Vector3dVector(chair_points)
chair_pcd.paint_uniform_color([0,0,1])

board_pcd = o3d.geometry.PointCloud()
board_pcd.points = o3d.utility.Vector3dVector(board_points)
board_pcd.paint_uniform_color([1, 0.5, 0])

# o3d.visualization.draw_geometries([chair_pcd, board_pcd])

mask = (board_points[:,0]>=1.768) & (board_points[:, 0] <= 2.1065)
filtered_board = board_points[mask]
filtered_board_centroid = np.mean(filtered_board, axis=0)

board_pcd.paint_uniform_color([0.6,0.6,0.6])
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_board)
filtered_pcd.paint_uniform_color([1, 0.5, 0])
# o3d.visualization.draw_geometries([chair_pcd, filtered_pcd])

diff = chair_centroid - filtered_board_centroid
distance = np.linalg.norm(diff)

distances = chair_pcd.compute_point_cloud_distance(filtered_pcd)
distances_np = np.asarray(distances)
print("의자 점 2,205개 각각에서 가장 가까운 칠판 점까지의 거리 배열: ", distances_np, "m")

min_distance = np.min(distances_np)
print("의자와 칠판 사이의 최소 점간(두개의 점) 거리: ", min_distance, "m") 

p5_distance = np.percentile(distances, 5)
print("최근접 거리의 5번째 백분위수: ", p5_distance)

percentiles = np.percentile(distances_np, [5, 25, 50, 75, 95])
# print(percentiles)

# 1. distances_np에서 최솟값의 인덱스를 찾아 min_idx에 저장
min_idx = np.argmin(distances_np)
# 2. min_idx로 의자 점의 XYZ 좌표를 찾아 closest_chair_point에 저장
closest_chair_point = chair_points[min_idx]
# 3. closest_chair_point와 모든 칠판 점의 XYZ 차이를 계산
closest_diff = closest_chair_point - filtered_board
# 4. 각 칠판 점까지의 3차원 거리를 계산
closest_distance = np.linalg.norm(closest_diff, axis=1)
# 5. 가장 작은 거리의 칠판 인덱스 찾기
closest_min_idx = np.argmin(closest_distance)
# 6. closest_board_point에 칠판 점의 XYZ 좌표 저장
closest_board_point = filtered_board[closest_min_idx]
# 7. 의자 점과 칠판 점의 XYZ 좌표 출력
print(closest_chair_point, closest_board_point)

# 1. 두 점을 하나의 (2, 3) 배열로 묶기
line_points = np.vstack([closest_chair_point, closest_board_point])

# 2. LineSet 객체 생성
line_set = o3d.geometry.LineSet()

# 3. line_points를 line_set.points에 저장
line_set.points = o3d.utility.Vector3dVector(line_points)

# 4. shape 확인
print(line_points.shape)
print(line_points)


# 1. 0번 점과 1번 점을 연결하도록 line_set.lines 설정
line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
# 2. 선 하나를 초록색 [0, 1, 0]으로 지정
line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
# 3. 의자, 칠판, 연결선을 한 창에 표시
# o3d.visualization.draw_geometries([chair_pcd, filtered_pcd, line_set])

plane_model, inliers = filtered_pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

board_normal_vector = np.array(plane_model[:3])
d = plane_model[3]

s = np.dot(board_normal_vector, filtered_board_centroid) + d

normal_length_squared = np.dot(board_normal_vector, board_normal_vector)

# 1. 칠판 법선을 의자가 있는 방향으로 정렬
board_surface_point = filtered_board_centroid - (s / normal_length_squared) * board_normal_vector
# 칠판 평면 위 기준점에서 의자 중심으로 향하는 벡터
direction = chair_centroid - board_surface_point

# 칠판 법선을 복사
board_normal_toward_chair = board_normal_vector.copy()

# 법선이 의자 반대쪽을 향하면 방향을 뒤집기
if np.dot(board_normal_toward_chair, direction) < 0:
    board_normal_toward_chair = -board_normal_toward_chair

# 법선의 길이를 1로 정규화
unit_normal = (
    board_normal_toward_chair
    / np.linalg.norm(board_normal_toward_chair)
)
# 2. 칠판 평면에서 1m 떨어진 목표점 계산

target_distance = 1.0

target_position = (
    board_surface_point
    + target_distance * unit_normal
)

print("\n[목표 위치]")
print("칠판 평면 위 기준점:", board_surface_point)
print("의자 쪽 단위 법선:", unit_normal)
print("목표 위치:", target_position)

# 3. 의자 전체를 목표 위치로 평행이동

# 목표 위치 - 현재 의자 중심 = 이동 벡터
translation = target_position - chair_centroid

# 원본 chair_points는 보존
moved_chair = chair_points + translation

# 이동된 의자의 중심
moved_chair_centroid = np.mean(moved_chair, axis=0)

print("\n[의자 이동]")
print("이동 벡터:", translation)
print("이동 전 중심:", chair_centroid)
print("이동 후 중심:", moved_chair_centroid)


# 4. 이동 후 의자 PointCloud 만들기

moved_chair_pcd = o3d.geometry.PointCloud()
moved_chair_pcd.points = o3d.utility.Vector3dVector(moved_chair)

moved_chair_pcd.paint_uniform_color([0, 1, 0])


# 5. 칠판 평면에서 의자 중심까지의 거리 검증

a, b, c, d = plane_model

# 이동한 의자 중심을 평면 방정식에 대입
plane_value = (
    np.dot(plane_model[:3], moved_chair_centroid)
    + d
)

# 평면과 점 사이 수직 거리
verified_distance = (
    abs(plane_value)
    / np.linalg.norm(plane_model[:3])
)

print("\n[거리 검증]")
print("목표 거리:", target_distance, "m")
print("이동 후 의자 중심의 평면 수직 거리:", verified_distance, "m")
print("중심 위치 오차:",
      np.linalg.norm(moved_chair_centroid - target_position), "m")


# 6. 의자 실제 점군에서 칠판 평면까지의 거리 확인

# 이동한 의자의 모든 점에서 평면까지의 수직 거리
chair_plane_distances = np.abs(
    moved_chair @ plane_model[:3] + d
) / np.linalg.norm(plane_model[:3])

print("의자 점들의 최소 평면 거리:",
      np.min(chair_plane_distances), "m")


# 7. Open3D 시각화

# 원본 의자: 파랑
chair_pcd.paint_uniform_color([0, 0, 1])

# 칠판: 주황
filtered_pcd.paint_uniform_color([1, 0.5, 0])

# 이동 의자: 초록
moved_chair_pcd.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([
    chair_pcd,
    filtered_pcd,
    moved_chair_pcd
])
