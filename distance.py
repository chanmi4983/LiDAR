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
# 2. 의자 중심을 기준으로 회전
angle_deg = 90.0  # 반대로 돌리려면 -90, 반 바퀴는 180
angle_rad = np.deg2rad(angle_deg)

cos_angle = np.cos(angle_rad)
sin_angle = np.sin(angle_rad)

# Z축을 중심으로 회전하는 행렬
rotation_matrix = np.array([
    [cos_angle, -sin_angle, 0],
    [sin_angle,  cos_angle, 0],
    [0,          0,         1]
])

# 중심을 원점으로 옮겨 회전하고, 원래 위치로 되돌리기
rotated_chair = (
    (chair_points - chair_centroid) @ rotation_matrix.T
    + chair_centroid
)

rotated_centroid = np.mean(rotated_chair, axis=0)


# 3. 회전한 의자를 칠판 정면에 정렬하고 간격 맞추기

# 회전한 점들로 평면 거리를 다시 계산
signed_distances = (
    rotated_chair - board_surface_point
) @ unit_normal

current_min_distance = np.min(signed_distances)

target_distance = 1.0
move_amount = target_distance - current_min_distance

# 의자 중심을 칠판 평면에 투영
center_distance = np.dot(
    rotated_centroid - board_surface_point,
    unit_normal
)

center_projection = (
    rotated_centroid - center_distance * unit_normal
)

# 평면과 나란하게 움직여 칠판 기준점 정면에 정렬
alignment_translation = (
    board_surface_point - center_projection
)

# 정면 정렬 이동 + 간격 조정 이동
translation = (
    alignment_translation + move_amount * unit_normal
)

# 회전된 의자에 이동 적용
moved_chair = rotated_chair + translation
moved_chair_centroid = np.mean(moved_chair, axis=0)

print("\n[회전 및 이동]")
print("회전 각도:", angle_deg, "도")
print("이동 벡터:", translation)
print("최종 의자 중심:", moved_chair_centroid)

# 4. 이동 후 간격 검증
moved_signed_distances = (
    moved_chair - board_surface_point
) @ unit_normal

verified_min_distance = np.min(moved_signed_distances)
verified_center_distance = np.dot(
    moved_chair_centroid - board_surface_point,
    unit_normal
)

print("\n[거리 검증]")
print("목표 최소 간격:", target_distance, "m")
print("이동 후 최소 간격:", verified_min_distance, "m")
print("이동 후 중심 거리:", verified_center_distance, "m")
print("최소 간격 오차:",
      abs(verified_min_distance - target_distance), "m")


# 5. 가장 가까운 의자 점과 칠판 평면 위 수선의 발
nearest_idx = np.argmin(moved_signed_distances)
nearest_chair_point = moved_chair[nearest_idx]

# 의자 점에서 법선 방향 거리를 빼면 평면 위에 도달
projected_board_point = (
    nearest_chair_point
    - verified_min_distance * unit_normal
)

gap_line = o3d.geometry.LineSet()
gap_line.points = o3d.utility.Vector3dVector([
    projected_board_point,
    nearest_chair_point
])
gap_line.lines = o3d.utility.Vector2iVector([[0, 1]])
gap_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])


# 6. 이동한 의자 점군 생성
moved_chair_pcd = o3d.geometry.PointCloud()
moved_chair_pcd.points = o3d.utility.Vector3dVector(
    moved_chair
)
moved_chair_pcd.paint_uniform_color([0, 1, 0])


# 7. 거리 선의 양 끝을 구체로 표시
def make_marker(position, color, radius=0.02):
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius
    )
    marker.compute_vertex_normals()
    marker.paint_uniform_color(color)
    marker.translate(position)
    return marker


chair_marker = make_marker(
    nearest_chair_point, [1, 0, 0]
)
plane_marker = make_marker(
    projected_board_point, [1, 0, 0]
)


# 8. 시각화
chair_pcd.paint_uniform_color([0, 0, 1])
filtered_pcd.paint_uniform_color([1, 0.5, 0])

o3d.visualization.draw_geometries(
    [
        chair_pcd,
        filtered_pcd,
        moved_chair_pcd,
        gap_line,
        chair_marker,
        plane_marker
    ],
    window_name="Chair-to-plane minimum gap: 1 m"
)