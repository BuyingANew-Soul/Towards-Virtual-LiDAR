import numpy as np
import open3d as o3d

def render_depth(pose, intrinsics, geometry):
    
    max_depth = 150.0
    
    # Perform raycasting if triangle mesh is provided
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        depth = raycasting(pose, intrinsics, geometry, max_depth)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        depth = projection(pose, intrinsics, geometry, max_depth)
        
    return depth
    

def raycasting(pose, intrinsics, mesh, max_depth):
    
    print("Creating depth map (raycasting)...")

    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_id = scene.add_triangles(mesh)
    
    # Render depths and normals for color camera
    rays_orig = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsics.intrinsic_matrix, np.eye(4), 
        intrinsics.width, intrinsics.height)
    V = (rays_orig[:,:,3:]).numpy()
    
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsics.intrinsic_matrix, np.linalg.inv(pose), 
            intrinsics.width, intrinsics.height)
            
    result = scene.cast_rays(rays)
    D = result['t_hit'].numpy()
    D_inf = D > max_depth
    
    N = result['primitive_normals'].numpy()
    R = np.linalg.inv(pose[:3,:3])
    N_t = np.zeros_like(N)
    N_t[:,:,0] = R[0,0]*N[:,:,0] + R[0,1]*N[:,:,1] + R[0,2]*N[:,:,2]
    N_t[:,:,1] = R[1,0]*N[:,:,0] + R[1,1]*N[:,:,1] + R[1,2]*N[:,:,2]
    N_t[:,:,2] = R[2,0]*N[:,:,0] + R[2,1]*N[:,:,1] + R[2,2]*N[:,:,2]

    N_invalid = (V[:,:,0]*N_t[:,:,0] + V[:,:,1]*N_t[:,:,1] + V[:,:,2]*N_t[:,:,2]) > 0
    D[D_inf | N_invalid] = 0
    
    return D
    
def projection(pose, intrinsics, pcd, max_depth):
    
    print("Creating depth map (projection)...")
    
    pcd_cam = pcd.transform(pose)
    points_cam = np.asarray(pcd_cam.points)
    
    # Filter out points behind the camera
    valid_mask = (points_cam[:, 2] > 0)
    points_cam = points_cam[valid_mask]

    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    width = intrinsics.width
    height = intrinsics.height

    # Projection
    u = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(np.int32)
    v = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(np.int32)
    z = points_cam[:, 2]

    # Filter out-of-bounds
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z <= max_depth)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

    # Create depth map (initialize with inf)
    D = np.full((height, width), np.inf, dtype=np.float32)
    # Use the nearest depth in case of collisions
    D[v, u] = np.minimum(D[v, u], z)
    D[D == np.inf] = 0.0 # Replace inf with 0
    
    return D
    
    