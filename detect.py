"""
LVX2 공간 점군에서 Chair.csv 견본과 비슷한 모양을 찾는 실험.

좌표 단위: m
방법: FPFH 특징 비교 → RANSAC 초기 정합 → ICP 정밀 정합
AI 학습 모델은 사용하지 않음.
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np


# --------------------------------------------------
# 1. 의자 견본 CSV 읽기
# --------------------------------------------------
def read_template(path):
    points = []

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # 장치 정보 행은 제외하고 점 데이터만 읽기
            if row.get("Data Type") not in ("1", "2"):
                continue

            points.append([
                float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),
            ])

    points = np.asarray(points, dtype=float)

    if len(points) < 10 or not np.isfinite(points).all():
        raise ValueError(
            "견본에는 유효한 XYZ 좌표가 최소 10개 필요합니다."
        )

    return points


# --------------------------------------------------
# 2. LVX2 파일에서 지정 시간 구간의 점군 읽기
# --------------------------------------------------
def read_lvx2(path, start=0.0, seconds=1.0):
    data = Path(path).read_bytes()

    # 파일 형식 확인
    if (
        len(data) < 29
        or data[:16] != b"livox_tech" + b"\0" * 6
        or data[16:20] != bytes([2, 0, 0, 0])
        or struct.unpack_from("<I", data, 20)[0] != 0xAC0EA767
    ):
        raise ValueError("지원하지 않는 LVX2 헤더입니다.")

    frame_duration, device_count = struct.unpack_from(
        "<IB", data, 24
    )

    if device_count != 1:
        raise ValueError(
            "이 프로그램은 라이다 1개로 기록한 파일만 지원합니다."
        )

    if len(data) < 29 + 63 * device_count:
        raise ValueError("장치 헤더가 잘렸습니다.")

    device = struct.unpack_from(
        "<16s16sIBBB6f", data, 29
    )

    if device[5] and any(
        abs(value) > 1e-8 for value in device[6:]
    ):
        raise ValueError(
            "장치 외부 보정값에 대한 좌표 변환이 필요합니다."
        )

    offset = 29 + 63 * device_count

    chunks = []
    first_timestamp = None
    last_timestamp = None

    frame_count = 0
    packet_count = 0
    total_points = 0

    while offset < len(data):
        if offset + 24 > len(data):
            raise ValueError("프레임 헤더가 잘렸습니다.")

        current_offset, next_offset, frame_index = (
            struct.unpack_from("<QQQ", data, offset)
        )

        if (
            current_offset != offset
            or next_offset <= offset + 24
            or next_offset > len(data)
        ):
            raise ValueError("프레임 연결 정보가 잘못되었습니다.")

        position = offset + 24
        frame_count += 1

        while position < next_offset:
            if position + 27 > next_offset:
                raise ValueError("패킷 헤더가 잘렸습니다.")

            (
                version,
                lidar_id,
                lidar_type,
                timestamp_type,
                timestamp,
                udp_count,
                data_type,
                payload_length,
                frame_counter,
                reserved,
            ) = struct.unpack_from(
                "<BIBBQHBIB4s", data, position
            )

            position += 27

            if (
                version != 0
                or data_type not in (1, 2)
                or position + payload_length > next_offset
            ):
                raise ValueError(
                    "지원하지 않거나 손상된 패킷입니다."
                )

            if data_type == 1:
                point_dtype = np.dtype([
                    ("xyz", "<i4", (3,)),
                    ("reflectivity", "u1"),
                    ("tag", "u1"),
                ])
                scale = 0.001
            else:
                point_dtype = np.dtype([
                    ("xyz", "<i2", (3,)),
                    ("reflectivity", "u1"),
                    ("tag", "u1"),
                ])
                scale = 0.01

            if payload_length % point_dtype.itemsize:
                raise ValueError(
                    "점 데이터 길이가 올바르지 않습니다."
                )

            if first_timestamp is None:
                first_timestamp = timestamp

            if (
                last_timestamp is not None
                and timestamp < last_timestamp
            ):
                raise ValueError(
                    "타임스탬프가 역행하는 파일입니다."
                )

            last_timestamp = timestamp
            packet_count += 1

            point_count = (
                payload_length // point_dtype.itemsize
            )
            total_points += point_count

            elapsed = (
                timestamp - first_timestamp
            ) / 1e9

            if start <= elapsed < start + seconds:
                packet_points = np.frombuffer(
                    data,
                    dtype=point_dtype,
                    count=point_count,
                    offset=position,
                )

                points = (
                    packet_points["xyz"].astype(float)
                    * scale
                )

                # 원점에 기록된 무효 점 제외
                valid = np.linalg.norm(
                    points, axis=1
                ) > 0

                points = points[valid]
                chunks.append(points)

            position += payload_length

        offset = next_offset

    if not chunks:
        raise ValueError(
            "요청한 시간 구간에 점 데이터가 없습니다."
        )

    metadata = {
        "frames": frame_count,
        "packets": packet_count,
        "total_points": total_points,
        "duration_s": (
            last_timestamp - first_timestamp
        ) / 1e9,
        "window_start_s": start,
        "window_seconds": seconds,
        "frame_duration_ms": frame_duration,
    }

    return np.concatenate(chunks), metadata


# --------------------------------------------------
# 3. NumPy 좌표 배열을 Open3D 점군으로 변환
# --------------------------------------------------
def cloud(points):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = (
        o3d.utility.Vector3dVector(points)
    )

    return point_cloud


# --------------------------------------------------
# 4. 점 수 줄이기 → 법선 추정 → FPFH 특징 계산
# --------------------------------------------------
def prepare(point_cloud, voxel):
    import open3d as o3d

    downsampled = point_cloud.voxel_down_sample(voxel)

    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 3,
            max_nn=40,
        )
    )

    features = (
        o3d.pipelines.registration.compute_fpfh_feature(
            downsampled,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 6,
                max_nn=100,
            ),
        )
    )

    return downsampled, features


# --------------------------------------------------
# 5. 견본과 비슷한 위치 찾기
# --------------------------------------------------
def match(scene, template, voxel=0.02, attempts=4):
    import open3d as o3d
    from scipy.spatial import cKDTree

    registration = o3d.pipelines.registration

    # 견본의 기존 위치를 검색에 이용하지 않도록
    # 중심을 빼서 원점 주변으로 이동
    centered = template - template.mean(axis=0)

    source, source_features = prepare(
        cloud(centered), voxel
    )

    target, target_features = prepare(
        cloud(scene), voxel
    )

    if len(source.points) < 10:
        raise ValueError(
            "점 수를 줄인 뒤 견본 점이 너무 적습니다."
        )

    # 공간 점군에서 가까운 점을 빠르게 검색하는 구조
    tree = cKDTree(np.asarray(target.points))

    candidates = []

    for seed in range(attempts):
        o3d.utility.random.seed(seed)

        # 특징을 비교해 대략적인 회전·이동 찾기
        initial = (
            registration
            .registration_ransac_based_on_feature_matching(
                source,
                target,
                source_features,
                target_features,
                True,
                voxel * 2.0,
                registration
                .TransformationEstimationPointToPoint(False),
                3,
                [
                    registration
                    .CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    registration
                    .CorrespondenceCheckerBasedOnDistance(
                        voxel * 2
                    ),
                ],
                registration.RANSACConvergenceCriteria(
                    150000,
                    0.999,
                ),
            )
        )

        # 초기 위치에서 ICP로 정밀하게 맞추기
        fine = registration.registration_icp(
            source,
            target,
            voxel * 1.5,
            initial.transformation,
            registration
            .TransformationEstimationPointToPoint(),
            registration.ICPConvergenceCriteria(
                max_iteration=100
            ),
        )

        transformation = fine.transformation

        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]

        # 각 행에 점 하나가 있으므로 R의 전치행렬 사용
        moved = (
            np.asarray(source.points) @ rotation.T
            + translation
        )

        # 옮긴 견본의 각 점에서 공간의 최근접점까지 거리
        distances = tree.query(moved)[0]

        valid = distances <= voxel * 1.5

        # 거리 조건을 통과한 견본 점 비율
        coverage = float(valid.mean())

        # 통과한 대응점의 거리 RMSE
        if valid.any():
            rmse = float(
                np.sqrt(
                    np.mean(distances[valid] ** 2)
                )
            )
        else:
            rmse = None

        candidates.append({
            "seed": seed,
            "coverage": coverage,
            "rmse_m": rmse,
            "transformation": transformation.tolist(),
        })

        print(
            "candidate",
            seed,
            "coverage",
            round(coverage, 3),
            "rmse_m",
            rmse,
            flush=True,
        )

    # 일치 비율이 높은 순서,
    # 같은 비율이면 거리 오차가 작은 순서
    candidates.sort(
        key=lambda candidate: (
            -candidate["coverage"],
            (
                candidate["rmse_m"]
                if candidate["rmse_m"] is not None
                else float("inf")
            ),
        )
    )

    best = candidates[0]

    plausible = (
        best["coverage"] >= 0.8
        and best["rmse_m"] is not None
        and best["rmse_m"] <= voxel
    )

    # 비슷한 점수인데 위치가 다른 후보가 있는지 확인
    centers = [
        np.asarray(candidate["transformation"])[:3, 3]
        for candidate in candidates
        if candidate["coverage"]
        >= best["coverage"] - 0.05
    ]

    ambiguous = any(
        np.linalg.norm(center - centers[0]) > 0.3
        for center in centers[1:]
    )

    if plausible and not ambiguous:
        status = "candidate_match"
    elif plausible:
        status = "ambiguous"
    else:
        status = "not_found"

    result = {
        "status": status,
        "best": best,
        "candidates": candidates,
        "template_downsampled_points": len(source.points),
        "scene_downsampled_points": len(target.points),
        "voxel_m": voxel,
    }

    return result, centered


# --------------------------------------------------
# 6. 파일 읽기 → 정합 → 저장 → 시각화
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("--scene", required=True)
    parser.add_argument("--template", required=True)

    parser.add_argument(
        "--start", type=float, default=0
    )
    parser.add_argument(
        "--seconds", type=float, default=1
    )
    parser.add_argument(
        "--voxel", type=float, default=0.02
    )
    parser.add_argument(
        "--attempts", type=int, default=4
    )

    parser.add_argument("--out", default="results")
    parser.add_argument("--view", action="store_true")

    args = parser.parse_args()

    if (
        args.seconds <= 0
        or args.start < 0
        or args.voxel <= 0
        or args.attempts < 1
    ):
        parser.error("숫자 옵션의 범위가 잘못되었습니다.")

    import open3d as o3d

    # 입력 파일 읽기
    scene, metadata = read_lvx2(
        args.scene,
        args.start,
        args.seconds,
    )

    template = read_template(args.template)

    print(
        "LVX2:",
        metadata,
        "\n공간 점 개수:",
        len(scene),
        "\n견본 점 개수:",
        len(template),
        flush=True,
    )

    # 형상 비교
    result, centered = match(
        scene,
        template,
        args.voxel,
        args.attempts,
    )

    result["recording"] = metadata
    result["template_points"] = len(template)

    result["limitations"] = (
        "Geometric template candidate only. "
        "Input correspondence is unverified; "
        "no independent recognition validation. "
        "No original template translation used in search."
    )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    # 전체 공간을 회색으로 표시
    scene_cloud = cloud(scene).voxel_down_sample(
        args.voxel
    )

    scene_cloud.paint_uniform_color(
        [0.55, 0.6, 0.65]
    )

    o3d.io.write_point_cloud(
        str(output / "scene.ply"),
        scene_cloud,
    )

    # 가장 좋은 후보 위치로 견본 이동
    transformation = np.asarray(
        result["best"]["transformation"]
    )

    matched_cloud = cloud(centered)
    matched_cloud.transform(transformation)

    matched_cloud.paint_uniform_color(
        [1.0, 0.15, 0.05]
    )

    o3d.io.write_point_cloud(
        str(output / "matched_template.ply"),
        matched_cloud,
    )

    # 평가 결과 저장
    (output / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # 전체 모습과 확대 모습을 이미지로 저장
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    scene_points = np.asarray(scene_cloud.points)
    matched_points = np.asarray(matched_cloud.points)

    random_generator = np.random.default_rng(0)

    selected_indices = random_generator.choice(
        len(scene_points),
        min(50000, len(scene_points)),
        replace=False,
    )

    sampled_points = scene_points[selected_indices]

    figure = plt.figure(figsize=(12, 5))

    for index in (1, 2):
        axis = figure.add_subplot(
            1, 2, index, projection="3d"
        )

        if index == 1:
            visible_points = sampled_points[
                np.linalg.norm(
                    sampled_points, axis=1
                ) < 8
            ]

            axis.set_title(
                "Scene and geometric candidate"
            )
        else:
            visible_points = scene_points[
                np.linalg.norm(
                    scene_points
                    - matched_points.mean(axis=0),
                    axis=1,
                ) < 0.65
            ]

            axis.set_title(
                "Candidate close-up "
                "(not validated recognition)"
            )

        axis.scatter(
            *visible_points.T,
            s=0.4,
            c="#8899aa",
            alpha=0.4,
        )

        axis.scatter(
            *matched_points.T,
            s=2,
            c="#ee4422",
        )

        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")

        axis.set_box_aspect((1, 1, 1))

    figure.tight_layout()

    figure.savefig(
        output / "preview.png",
        dpi=160,
    )

    plt.close(figure)

    print(json.dumps(result, indent=2))

    # --view를 붙였을 때 3D 창 열기
    # 상태와 관계없이 최상위 후보를 진단용으로 표시
    if args.view:
        box = matched_cloud.get_oriented_bounding_box()
        box.color = (1, 0, 0)

        o3d.visualization.draw_geometries([
            scene_cloud,
            matched_cloud,
            box,
        ])


if __name__ == "__main__":
    main()