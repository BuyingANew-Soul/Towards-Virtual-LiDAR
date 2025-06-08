# Virtual LiDAR: Depth Estimation Using Aerial Laser Scans

## This repository implements a system for generating depth maps from a ground vehicle's perspective by combining aerial laser scanning (ALS) data with ground-level camera imagery. The goal is to simulate LiDAR depth perception (i.e., "Virtual LiDAR") for use in autonomous driving research, offering a low-cost alternative to onboard LiDAR sensors.

---

## 🧠 Key Features

- Fuse aerial point cloud data with camera intrinsics and poses.
- Render depth maps using:
  - Raycasting (for triangle meshes)
  - Direct projection (for point clouds)
- Support for multiple surface reconstruction methods:
  - Delaunay triangulation (proposed method)
  - Poisson reconstruction
  - Ball Pivoting Algorithm (BPA)
  - Alpha Shapes
- Evaluation against ground-truth LiDAR scans
- Visualization of depth maps and error overlays

---

