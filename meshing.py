import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator

def pcd_to_depth_ortho(cloud, extrinsics, width, height):
    """ Create depth map from point cloud (orthographic projection) """
    points = np.asarray(cloud.points)  # shape (N, 3)
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    points_cam = (extrinsics @ points_hom.T).T  # (N, 4)
    points_cam = points_cam[:, :3]  # Keep x, y, z
    valid_mask = (points_cam[:, 2] > 0)
    points_cam = points_cam[valid_mask]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    scale = np.amax(z) / height
    u = (x / scale + width / 2).astype(np.int32)
    v = (y / scale + height / 2).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    depth_map[v, u] = np.minimum(depth_map[v, u], z)
    depth_map[depth_map == np.inf] = 0.0
    return depth_map

def depth_to_mesh(depth, extrinsics=None):
    """ Convert depth into a triangle mesh via Delaunay triangulation. """
    height, width = depth.shape
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    if len(z) == 0:
        print("Error: No valid depth points for Delaunay triangulation")
        return o3d.geometry.TriangleMesh()
    scale = np.max(z) / height
    x = (u - width / 2) * scale
    y = (v - height / 2) * scale
    vertices = np.stack((x, y, z), axis=1)
    pts2d = np.stack((u, v), axis=1)
    print("Running Delaunay...")
    tri = Delaunay(pts2d)
    triangles = tri.simplices[:, [0, 2, 1]]  # ensure correct winding
    print("Finished!")
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    if extrinsics is not None:
        mesh = mesh.transform(extrinsics)
    return mesh

def create_mesh_poisson(pcd, depth_param=8):
    """Create mesh using Poisson reconstruction directly from point cloud"""
    print(f"Running Poisson reconstruction directly on point cloud (depth={depth_param})...")
    
    # Estimate normals
    points = np.asarray(pcd.points)
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(points)
    
    # Adaptive radius based on point cloud size
    bbox = pcd_copy.get_axis_aligned_bounding_box()
    bbox_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
    radius = bbox_size * 0.01  # 1% of bounding box diagonal
    max_nn = min(50, len(points) // 10)
    
    pcd_copy.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd_copy.orient_normals_consistent_tangent_plane(k=max_nn)
    
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_copy, depth=depth_param, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low density vertices
        if len(densities) > 0:
            density_threshold = np.quantile(np.asarray(densities), 0.01)
            vertices_to_remove = np.where(np.asarray(densities) < density_threshold)[0]
            mesh.remove_vertices_by_index(vertices_to_remove)
        
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        print("Finished Poisson reconstruction!")
        return mesh
        
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        return o3d.geometry.TriangleMesh()

def create_mesh_alpha(pcd, alpha_factor=2.0):
    """Create mesh using Alpha Shapes directly from point cloud"""
    print(f"Running Alpha Shapes directly on point cloud...")
    
    points = np.asarray(pcd.points)
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(points)
    
    # Calculate alpha based on average nearest neighbor distance
    distances = pcd_copy.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances) if len(distances) > 0 else 0.1
    alpha = alpha_factor * avg_dist
    
    print(f"Using alpha={alpha:.4f}...")
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_copy, alpha=alpha)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        print("Finished Alpha Shapes!")
        return mesh
        
    except Exception as e:
        print(f"Alpha Shapes reconstruction failed: {e}")
        return o3d.geometry.TriangleMesh()

def create_mesh_bpa(pcd, radius_factor=2.0):
    """Create mesh using Ball Pivoting Algorithm directly from point cloud"""
    print(f"Running Ball Pivoting Algorithm directly on point cloud...")
    
    points = np.asarray(pcd.points)
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    bbox = pcd_copy.get_axis_aligned_bounding_box()
    bbox_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
    radius = bbox_size * 0.01
    max_nn = min(50, len(points) // 10)
    
    pcd_copy.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd_copy.orient_normals_consistent_tangent_plane(k=max_nn)
    
    # Calculate radii for ball pivoting
    distances = pcd_copy.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances) if len(distances) > 0 else 0.1
    radii = [radius_factor * avg_dist, 2 * radius_factor * avg_dist]
    
    print(f"Using radii: {radii}")
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_copy, o3d.utility.DoubleVector(radii)
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        print("Finished Ball Pivoting!")
        return mesh
        
    except Exception as e:
        print(f"Ball Pivoting reconstruction failed: {e}")
        return o3d.geometry.TriangleMesh()

def filter_depth(depth, diameter=5, sigma_color=2, sigma_space=10):
    h, w = depth.shape
    y, x = np.indices((h, w))
    valid = depth > 1e-3
    coords = np.column_stack((x[valid], y[valid]))
    values = depth[valid]
    xy_min = np.amin(coords, axis=0)
    xy_max = np.amax(coords, axis=0)
    x0, y0 = xy_min
    x1, y1 = xy_max
    print("Filling holes...")
    interpolator = NearestNDInterpolator(coords, values)
    filled_depth = interpolator(x, y)
    print("Filtering depth map...")
    filled_depth_32f = filled_depth.astype(np.float32)
    filtered_depth = cv2.bilateralFilter(filled_depth_32f, diameter, sigma_color, sigma_space)
    print("Finished!")
    mask = np.zeros_like(depth, dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True
    filtered_depth[~mask] = 0.0
    return filtered_depth

def visualize_depth(depth, title):
    valid = depth > 0
    d_min = np.min(depth[valid]) if np.any(valid) else 0
    d_max = np.max(depth[valid]) if np.any(valid) else 1
    depth_vis = np.zeros_like(depth, dtype=np.float32)
    depth_vis[valid] = (depth[valid] - d_min) / (d_max - d_min) if d_max > d_min else depth[valid]
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_vis, cmap='turbo', interpolation='nearest')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# def compute_mesh(pcd, method="delaunay", visualize=False, **kwargs):
#     width, height = 2000, 2000
#     print(f"Computing mesh from the aerial point cloud using {method}...")
#     if not pcd.has_points():
#         print("Error: No points found in the point cloud.")
#         return o3d.geometry.TriangleMesh()
#     plane_model, inliers = pcd.segment_plane(distance_threshold=1.0, ransac_n=3, num_iterations=1000)
#     normal = np.array(plane_model[:3])
#     normal /= np.linalg.norm(normal)
#     d = plane_model[3]
#     print(f"Plane equation: {normal[0]:.2f}x + {normal[1]:.2f}y + {normal[2]:.2f}z + {d:.2f} = 0")
#     center = np.mean(np.asarray(pcd.points), axis=0)
#     camera_pos = center + 1000 * normal
#     z_axis = (center - camera_pos)
#     z_axis /= np.linalg.norm(z_axis)
#     world_up = np.array([0, 1, 0]) if abs(np.dot(z_axis, [0, 1, 0])) < 0.99 else np.array([1, 0, 0])
#     x_axis = np.cross(world_up, z_axis)
#     x_axis /= np.linalg.norm(x_axis)
#     y_axis = np.cross(z_axis, x_axis)
#     R = np.stack([x_axis, y_axis, z_axis], axis=1)
#     R_wc = R.T
#     t_wc = -R_wc @ camera_pos
#     extrinsics = np.hstack((R_wc, t_wc.reshape(3, 1)))
#     extrinsics = np.vstack([extrinsics, [0, 0, 0, 1]])
#     depth_sparse = pcd_to_depth_ortho(pcd, extrinsics, width, height)
#     depth_filt = filter_depth(depth_sparse)
#     if visualize:
#         visualize_depth(depth_sparse, title="Sparse Depth")
#         visualize_depth(depth_filt, title="Filtered Depth")
#     if method == "delaunay":
#         mesh = depth_to_mesh(depth_filt, extrinsics=np.linalg.inv(extrinsics))
#     elif method == "poisson":
#         mesh = depth_to_mesh_poisson(depth_filt, extrinsics=np.linalg.inv(extrinsics), depth_param=kwargs.get("depth_param", 8))
#     elif method == "alpha_shapes":
#         mesh = depth_to_mesh_alpha(depth_filt, extrinsics=np.linalg.inv(extrinsics), alpha_factor=kwargs.get("alpha_factor", 2.0))
#     elif method == "ball_pivoting":
#         mesh = depth_to_mesh_bpa(depth_filt, extrinsics=np.linalg.inv(extrinsics), radius_factor=kwargs.get("radius_factor", 2.0))
#     else:
#         raise ValueError(f"Unknown method: {method}")
#     if visualize:
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
#         camera_frame.rotate(R, center=np.zeros(3))
#         camera_frame.translate(camera_pos)
#         o3d.visualization.draw_geometries(
#             [mesh, camera_frame],
#             zoom=0.6,
#             front=-z_axis,
#             lookat=center,
#             up=y_axis
#         )
#     return mesh

def compute_mesh(pcd, method="delaunay", visualize=False, **kwargs):
    """
    Compute mesh using different methods - corrected version
    """
    print(f"Computing mesh from the aerial point cloud using {method}...")
    
    if not pcd.has_points():
        print("Error: No points found in the point cloud.")
        return o3d.geometry.TriangleMesh()
    
    if method == "delaunay":
        # For Delaunay, we still need the depth map approach since it works on 2D
        width, height = 2000, 2000
        
        # Plane fitting and camera setup (same as before)
        plane_model, inliers = pcd.segment_plane(distance_threshold=1.0, ransac_n=3, num_iterations=1000)
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        d = plane_model[3]
        
        center = np.mean(np.asarray(pcd.points), axis=0)
        camera_pos = center + 1000 * normal
        z_axis = (center - camera_pos)
        z_axis /= np.linalg.norm(z_axis)
        world_up = np.array([0, 1, 0]) if abs(np.dot(z_axis, [0, 1, 0])) < 0.99 else np.array([1, 0, 0])
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        R_wc = R.T
        t_wc = -R_wc @ camera_pos
        extrinsics = np.hstack((R_wc, t_wc.reshape(3, 1)))
        extrinsics = np.vstack([extrinsics, [0, 0, 0, 1]])
        
        # Create depth map and mesh
        depth_sparse = pcd_to_depth_ortho(pcd, extrinsics, width, height)
        depth_filt = filter_depth(depth_sparse)
        
        if visualize:
            visualize_depth(depth_sparse, title="Sparse Depth")
            visualize_depth(depth_filt, title="Filtered Depth")
        
        mesh = depth_to_mesh(depth_filt, extrinsics=np.linalg.inv(extrinsics))
        
    elif method == "poisson":
        # Use original point cloud directly
        mesh = create_mesh_poisson(pcd, depth_param=kwargs.get("depth_param", 8))
        
    elif method == "alpha_shapes":
        # Use original point cloud directly
        mesh = create_mesh_alpha(pcd, alpha_factor=kwargs.get("alpha_factor", 2.0))
        
    elif method == "ball_pivoting":
        # Use original point cloud directly
        mesh = create_mesh_bpa(pcd, radius_factor=kwargs.get("radius_factor", 2.0))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if visualize and method == "delaunay":
        # Visualization code for delaunay (same as before)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        camera_frame.rotate(R, center=np.zeros(3))
        camera_frame.translate(camera_pos)
        o3d.visualization.draw_geometries(
            [mesh, camera_frame],
            zoom=0.6,
            front=-z_axis,
            lookat=center,
            up=y_axis
        )
    
    return mesh
