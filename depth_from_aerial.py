import os
import sys
import cv2
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import stats

import utils.io_utils as io
from meshing import compute_mesh
from rendering import render_depth

# python depth_from_aerial.py input/images/image_1556192549.936944088.png --output_dir='output/image_'
# python depth_from_aerial.py input/images/image_1556192505.567082551.png --output_dir='output/image_'
# python depth_from_aerial.py input/images/image_1556192528.268712736.png --output_dir='output/image_'

def compute_metrics(depth_pred, depth_gt):
    """ Compute evaluation metrics for depth map comparison - corrected version """
    
    # Valid pixels for each depth map
    pred_valid = depth_pred > 1e-3
    gt_valid = depth_gt > 1e-3
    both_valid = pred_valid & gt_valid  # pixels where both have valid depth
    
    # Total number of pixels
    total_pixels = depth_pred.shape[0] * depth_pred.shape[1]
    
    if not np.any(both_valid):
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "MAPE": np.nan,
            "δ_1.25": np.nan,
            "δ_1.5625": np.nan,
            "AbsAcc_1m": np.nan,
            "AbsAcc_2m": np.nan,
            "Completeness": np.sum(pred_valid) / total_pixels * 100,  # Still compute this
            "Coverage": 0.0  # No overlap between pred and GT
        }
    
    # Error calculations (only where both are valid)
    abs_error = np.abs(depth_pred[both_valid] - depth_gt[both_valid])
    
    # MAE (Mean Absolute Error)
    mae = np.mean(abs_error)
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(abs_error ** 2))
    
    # MedAE (Median Absolute Error)
    medae = np.median(abs_error)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(abs_error / depth_gt[both_valid]) * 100  # in percentage
    
    # δ_1.25 and δ_1.5625 (threshold accuracy)
    ratio = np.maximum(depth_pred[both_valid] / depth_gt[both_valid], 
                      depth_gt[both_valid] / depth_pred[both_valid])
    delta_1_25 = np.mean(ratio < 1.25) * 100  # in percentage
    delta_1_5625 = np.mean(ratio < 1.5625) * 100  # in percentage
    
    # AbsAcc_1m and AbsAcc_2m (absolute accuracy thresholds)
    abs_acc_1m = np.mean(abs_error < 1.0) * 100  # in percentage
    abs_acc_2m = np.mean(abs_error < 2.0) * 100  # in percentage
    
    # CORRECTED Completeness: percentage of ALL pixels with valid depth
    completeness = np.sum(pred_valid) / total_pixels * 100  # in percentage
    
    # Coverage: percentage of ground truth pixels that are covered by prediction
    coverage = np.sum(both_valid) / np.sum(gt_valid) * 100 if np.sum(gt_valid) > 0 else 0.0
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MedAE": medae,
        "MAPE": mape,
        "δ_1.25": delta_1_25,
        "δ_1.5625": delta_1_5625,
        "AbsAcc_1m": abs_acc_1m,
        "AbsAcc_2m": abs_acc_2m,
        "Completeness": completeness,  # % of all pixels with valid depth
        "Coverage": coverage  # % of GT pixels covered by prediction
    }

def generate_metrics_report(metrics, method_params, outpath):
    """ Generate a text report of evaluation metrics """
    report = "Surface Reconstruction Evaluation Report\n"
    report += "=" * 40 + "\n\n"
    
    for method, params in method_params.items():
        report += f"Method: {method}\n"
        report += "Parameters:\n"
        for param, value in params.items():
            if param != "method":  # Skip the method name itself
                report += f"  {param}: {value}\n"
        report += "Metrics:\n"
        for metric, value in metrics[method].items():
            report += f"  {metric}: {value:.4f}\n"
        report += "\n"
    
    with open(outpath, 'w') as f:
        f.write(report)
    print(f"Metrics report saved to {outpath}")

def visualize_depth_maps(color_img, depth_gt, depth_maps, titles, outpath):
    """ Visualize depth maps with color image and ground truth """
    fig = plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 14})  # Increase font size for readability
    
    # Color image (top left)
    plt.subplot(2, 3, 1)
    plt.imshow(color_img)
    plt.axis("off")
    plt.title("Color Image")
    
    # Ground truth depth (bottom left)
    valid = depth_gt > 0
    depth_vis = np.zeros_like(depth_gt, dtype=np.float32)
    if np.any(valid):
        d_min, d_max = np.min(depth_gt[valid]), np.max(depth_gt[valid])
        depth_vis[valid] = (depth_gt[valid] - d_min) / (d_max - d_min)
    plt.subplot(2, 3, 4)
    plt.imshow(depth_vis, cmap="turbo", interpolation='nearest')
    plt.axis("off")
    plt.title("Ground Truth Depth")
    
    # Depth maps from reconstruction methods
    for i, (depth, title) in enumerate(zip(depth_maps, titles), start=2):
        valid = depth > 0
        depth_vis = np.zeros_like(depth, dtype=np.float32)
        if np.any(valid):
            d_min, d_max = np.min(depth[valid]), np.max(depth[valid])
            depth_vis[valid] = (depth[valid] - d_min) / (d_max - d_min)
        plt.subplot(2, 3, i if i <= 3 else i + 1)
        plt.imshow(depth_vis, cmap="turbo", interpolation='nearest')
        plt.axis("off")
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()

def visualize_error_maps(color_img, depth_gt, depth_maps, titles, outpath):
    """ Visualize error maps for all reconstruction methods overlaid on the camera image """
    plt.rcParams.update({'font.size': 14})
    for depth, title in zip(depth_maps, titles):
        # Compute error map
        error = np.abs(depth - depth_gt)
        valid = (depth > 1e-3) & (depth_gt > 1e-3)
        error_vis = np.zeros_like(error, dtype=np.float32)
        if np.any(valid):
            e_max = np.percentile(error[valid], 95)
            error_vis[valid] = np.clip(error[valid] / e_max, 0, 1)
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        # Plot color image
        plt.imshow(color_img)
        # Overlay error map with transparency
        plt.imshow(error_vis, cmap="inferno", interpolation='nearest', alpha=0.5)
        plt.axis("off")
        plt.title(f"Error Map ({title.split('(')[1].split(')')[0]})")
        plt.colorbar(label="Normalized Error")
        plt.tight_layout()
        
        # Save individual figure
        method_name = title.split('(')[1].split(')')[0].lower().replace(" ", "_")
        save_path = os.path.join(os.path.dirname(outpath), f"error_map_{method_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create depth map from aerial scan.")
    parser.add_argument("image_path", type=str, help="Path to image file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    args = parser.parse_args()
        
    # Define input paths
    poses_path = os.path.join("input", "slam_poses.csv")
    calib_path = os.path.join("input", "calibration.json")
    aerial_scan_path = os.path.join("input", "aerial_scan", "alinged_ls_short.pcd")
    velodyne_dir = os.path.join("input", "velodyne_scans")
    
    os.makedirs(args.output_dir, exist_ok=True) 
    
    """ Load color image and undistort """
    
    # Load camera parameters
    width, height, K_init, dists, T_lidar2cam = io.load_calibration(calib_path)
    
    # Compute new camera matrix
    K, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix=K_init,
        distCoeffs=dists,
        imageSize=(width, height),
        alpha=0, # no black border
        newImgSize=(width, height),
        centerPrincipalPoint=True
    )
    print("New intrinsics")
    print(K)
    
    # Load and undistort color image
    assert os.path.exists(args.image_path), "Image not found!"
    color_img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.undistort(color_img, K_init, dists, None, K)
    
    # Extract timestamp from color image filename (assume 'image_<timestamp>.png')
    image_basename = os.path.basename(args.image_path)
    stamp_image_str = image_basename.replace('image_', '').replace('.png', '')
    stamp_image = float(stamp_image_str)
    print(f"Image timestamp: {stamp_image}")
    
    """ Load LiDAR points, camera pose, and aerial scan """
    
    pcd_lidar, stamp_pcd = io.load_velodyne(velodyne_dir, timestamp=stamp_image)
    stamp_pcd_dt = stamp_pcd - stamp_image
    print(f"Velodyne timestamp: {stamp_pcd:.8f} (dt = {stamp_pcd_dt:.4f})")
    
    assert os.path.exists(poses_path), "Camera poses not found!"
    T_base2map, stamp_pose = io.load_pose(poses_path, timestamp=stamp_image)
    stamp_pose_dt = stamp_pose - stamp_image
    print(f"Pose timestamp: {stamp_pose:.8f} (dt = {stamp_pose_dt:.4f})")
    
    assert os.path.exists(aerial_scan_path), "Aerial scan not found!"
    pcd_aerial = io.load_aerial_scan(aerial_scan_path)
    pcd_path = os.path.join(args.output_dir, "aerial_pcd.ply")
    o3d.io.write_point_cloud(pcd_path, pcd_aerial)
    
    """ Compute meshes from the aerial scan """
    
    reconstruction_methods = [
        ("proposed method", {"method": "delaunay"}),
        ("ball_pivoting", {"method": "ball_pivoting", 'radius_factor':3}),
        ("poisson", {"method": "poisson", "depth_param": 10}),
        ("alpha_shapes", {"method": "alpha_shapes", "alpha_factor": 3.0})
    ]
    
    meshes = {}
    for name, kwargs in reconstruction_methods:
        mesh = compute_mesh(pcd_aerial, visualize=False, **kwargs)
        meshes[name] = mesh
        mesh_path = os.path.join(args.output_dir, f"aerial_mesh_{name}.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
    
    """ Render depth maps """
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )
    
    T_lidar2base = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Pose (map -> base -> lidar -> camera)
    pose = T_lidar2cam @ np.linalg.inv(T_lidar2base) @ np.linalg.inv(T_base2map)
    pose = np.linalg.inv(pose)
    
    depth_maps = {}
    for name, mesh in meshes.items():
        depth = render_depth(pose, intrinsics, mesh)
        depth_maps[name] = depth
    
    """ Generate ground truth depth from LiDAR """
    
    pose = np.eye(4)
    pcd_lidar.transform(T_lidar2cam)
    depth_gt = render_depth(pose, intrinsics, pcd_lidar)
    depth_gt_vis = cv2.dilate(depth_gt, np.ones((5, 5), np.float32))
    
    """ Compute metrics """
    
    metrics = {}
    for name, depth in depth_maps.items():
        metrics[name] = compute_metrics(depth, depth_gt)
        print(f"\nMetrics for {name}:")
        for metric, value in metrics[name].items():
            print(f"{metric}: {value:.4f}")
    
    # Generate metrics report
    generate_metrics_report(metrics, dict(reconstruction_methods), os.path.join(args.output_dir, "metrics_report.txt"))
    
    """ Visualization """
    
    # Depth maps figure
    depth_map_titles = [f"Depth ({name})" for name in depth_maps.keys()]
    visualize_depth_maps(
        color_img,
        depth_gt_vis,
        list(depth_maps.values()),
        depth_map_titles,
        os.path.join(args.output_dir, "depth_maps.png")
    )
    
    # Error maps figure
    visualize_error_maps(
        color_img,
        depth_gt_vis,
        list(depth_maps.values()),
        depth_map_titles,
        os.path.join(args.output_dir, "error_maps.png")
    )
