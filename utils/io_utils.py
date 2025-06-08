import os
import sys
import json
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation

def load_calibration(calib_file):
    
    """ Load camera parameters from JSON file. """
    
    if not os.path.exists(calib_file):
        print(f"Calibration file not found: {calib_file}")
        sys.exit(1)

    with open(calib_file, "r") as f:
        calib_data = json.load(f)

    width = calib_data["width"]
    height = calib_data["height"]
    K = np.array(calib_data["intrinsic_matrix"], dtype=np.float64)
    K = K.reshape((3,3))
    dists = np.array(calib_data["distortion_coefficients"], dtype=np.float64)
    ext = np.array(calib_data["extrinsic_matrix"], dtype=np.float64)
    ext = ext.reshape((3,4))
    T_lidar2cam = np.eye(4)
    T_lidar2cam[:3, :] = ext
    
    print(f"Intrinsics ({height} x {width})")
    print(K)
    print(f"Distortion coefficients")
    print(dists)
    print(f"Extrinsics (LiDAR to camera")
    print(T_lidar2cam)
    
    return width, height, K, dists, T_lidar2cam
    
def load_velodyne(velodyne_dir, timestamp):
    
    """ Load point cloud (.pcd) with the closest timestamp. """

    target_ns = int(timestamp * 1e6)  # Convert to microseconds

    closest_file = None
    min_diff = float("inf")
    closest_stamp = None

    for fname in os.listdir(velodyne_dir):
        if fname.startswith("scan") and fname.endswith(".pcd"):
            try:
                ts_ns = int(fname[len("scan"):-len(".pcd")])
                diff = abs(ts_ns - target_ns)
                if diff < min_diff:
                    min_diff = diff
                    closest_file = fname
                    closest_stamp = float(ts_ns) / 1e6
            except ValueError:
                print("Invalid PCD!")
                continue  # Skip malformed filenames

    if closest_file is None:
        raise FileNotFoundError("No valid scan*.pcd files found.")

    pcd_path = os.path.join(velodyne_dir, closest_file)
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded Velodyne points: {closest_file}")

    return pcd, closest_stamp
    
def load_pose(poses_path, timestamp):
    
    """ Load camera pose with the closest timestamp from CSV file. """
    
    pose_df = pd.read_csv(poses_path, dtype=str)  # Read as strings to avoid float issues

    pose_dict = {
        float(row["timestamp"]): {
            "t": np.array([float(row["t_x"]), float(row["t_y"]), float(row["t_z"])]),
            "q": np.array([float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])]),
        }
        for _, row in pose_df.iterrows()
    }
    
    if not pose_dict:
        raise ValueError(f"❌ No poses found in {poses_path}")

    # Find the closest timestamp key
    closest_stamp = min(pose_dict.keys(), key=lambda ts: abs(ts - timestamp))
    pose = pose_dict[closest_stamp]

    t = pose["t"]
    q = pose["q"]
    
    # Convert quaternion to rotation matrix
    R = Rotation.from_quat(q).as_matrix()
    
    # Pose matrix
    pose_base2map = np.eye(4)
    pose_base2map[:3,:3] = R
    pose_base2map[:3,3] = t

    return pose_base2map, closest_stamp
    
def load_aerial_scan(aerial_scan_path):
    
    pcd_aerial = o3d.io.read_point_cloud(aerial_scan_path)
    pcd_aerial.paint_uniform_color([0.2, 0.2, 0.8])
    print(f"Loaded aerial scan: {aerial_scan_path}")
    
    return pcd_aerial
