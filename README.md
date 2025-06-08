Depth from Aerial Scans
This repository contains Python scripts for generating depth maps from aerial point cloud data and comparing them against ground truth depth maps derived from LiDAR scans. The project supports multiple mesh reconstruction methods, including Delaunay triangulation, Poisson reconstruction, Alpha Shapes, and Ball Pivoting Algorithm (BPA), and provides evaluation metrics and visualizations for the generated depth maps.
The codebase leverages the open3d library for 3D data processing, numpy for numerical computations, matplotlib and cv2 for visualization, and scipy for spatial operations like Delaunay triangulation and interpolation.
Features

Depth Map Generation: Generate depth maps from aerial point clouds using raycasting (for meshes) or projection (for point clouds).
Mesh Reconstruction: Supports four reconstruction methods:
Delaunay triangulation (from depth maps)
Poisson reconstruction
Alpha Shapes
Ball Pivoting Algorithm (BPA)


Evaluation Metrics: Compute comprehensive metrics (MAE, RMSE, MedAE, MAPE, δ_1.25, δ_1.5625, AbsAcc_1m, AbsAcc_2m, Completeness, Coverage) to compare predicted depth maps against ground truth.
Visualization: Generate visualizations of depth maps and error maps overlaid on the input color image.
Camera Calibration and Pose Handling: Undistort images and transform point clouds using camera intrinsics and extrinsics from provided calibration and pose data.

Folder Structure
├── input/
│   ├── images/                 # Input color images
│   ├── velodyne_scans/        # LiDAR scan data
│   ├── aerial_scan/           # Aerial point cloud data
│   ├── calibration.json       # Camera calibration parameters
│   └── slam_poses.csv         # Camera pose data
├── utils/
│   ├── io_utils.py            # Utility functions for loading data
│   └── __pycache__/           # Python cache files
├── depth_from_aerial.py       # Main script for depth map generation and evaluation
├── rendering.py               # Functions for rendering depth maps
├── meshing.py                 # Functions for mesh reconstruction
└── README.md                  # This file

Prerequisites
To run the code, ensure you have the following Python packages installed:
pip install numpy open3d matplotlib opencv-python scipy


Python: 3.8 or higher
Operating System: Tested on Linux and macOS; should work on Windows with minor adjustments.

Usage
The main script, depth_from_aerial.py, processes an input image, associated LiDAR data, and an aerial point cloud to generate depth maps, reconstruct meshes, and evaluate results.
Command-Line Arguments
python depth_from_aerial.py <image_path> --output_dir=<output_directory>


<image_path>: Path to the input color image (e.g., input/images/image_1556192549.936944088.png).
--output_dir: Directory to save output files (default: output).

Example
python depth_from_aerial.py input/images/image_1556192549.936944088.png --output_dir=output/image_

This command:

Loads the specified image, camera calibration, poses, and aerial point cloud.
Generates meshes using Delaunay, Poisson, Alpha Shapes, and BPA methods.
Renders depth maps from the meshes.
Computes evaluation metrics against ground truth depth from LiDAR.
Saves results (meshes, depth maps, error maps, and metrics report) to the specified output directory.

Input Requirements

Images: Color images in PNG format, named as image_<timestamp>.png.
Calibration: calibration.json containing camera intrinsics, distortion coefficients, and LiDAR-to-camera transformation.
Poses: slam_poses.csv with camera poses and timestamps.
LiDAR Scans: Velodyne scans in the velodyne_scans directory.
Aerial Scan: A point cloud file (alinged_ls_short.pcd) in the aerial_scan directory.

Output Files

Meshes: Saved as .ply files (e.g., aerial_mesh_proposed_method.ply).
Depth Maps: Visualized and saved as depth_maps.png.
Error Maps: Visualized and saved as error_map_<method>.png.
Metrics Report: Saved as metrics_report.txt with evaluation metrics for each method.

Key Components
1. depth_from_aerial.py
The main script that orchestrates the pipeline:

Loads and undistorts the input image.
Loads LiDAR points, camera poses, and aerial point cloud.
Generates meshes using meshing.py.
Renders depth maps using rendering.py.
Computes metrics and generates visualizations.

2. rendering.py
Handles depth map generation:

Raycasting: For triangle meshes using Open3D's RaycastingScene.
Projection: For point clouds using pinhole camera projection.
Filters invalid depths and applies normal-based culling for raycasting.

3. meshing.py
Implements mesh reconstruction methods:

Delaunay Triangulation: Projects point cloud to a depth map and triangulates in 2D.
Poisson Reconstruction: Uses Open3D's Poisson surface reconstruction.
Alpha Shapes: Constructs a mesh using alpha shapes.
Ball Pivoting Algorithm: Reconstructs surfaces using BPA.
Includes depth map filtering and visualization utilities.

4. utils/io_utils.py
Provides helper functions for loading:

Camera calibration (calibration.json).
Camera poses (slam_poses.csv).
Velodyne LiDAR scans.
Aerial point cloud (alinged_ls_short.pcd).

Evaluation Metrics
The script computes the following metrics for each reconstruction method:

MAE: Mean Absolute Error.
RMSE: Root Mean Square Error.
MedAE: Median Absolute Error.
MAPE: Mean Absolute Percentage Error.
δ_1.25, δ_1.5625: Percentage of pixels with depth ratio < 1.25 or 1.5625.
AbsAcc_1m, AbsAcc_2m: Percentage of pixels with absolute error < 1m or 2m.
Completeness: Percentage of pixels with valid depth.
Coverage: Percentage of ground truth pixels covered by the predicted depth.

Visualizations

Depth Maps: Shows the color image, ground truth depth, and predicted depth maps for all methods.
Error Maps: Overlays error maps (difference between predicted and ground truth depth) on the color image.

Notes

The aerial point cloud should be pre-aligned (e.g., alinged_ls_short.pcd).
The timestamp in the image filename must match closely with pose and LiDAR scan timestamps for accurate alignment.
Adjust parameters (e.g., depth_param, alpha_factor, radius_factor) in reconstruction_methods to fine-tune mesh quality.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using Open3D for 3D processing.
Utilizes SciPy for Delaunay triangulation and interpolation.
Visualization powered by Matplotlib and OpenCV.

For issues or contributions, please open a pull request or issue on this GitHub repository.
