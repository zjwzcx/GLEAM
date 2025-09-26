import numpy as np
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as etxml
import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
    # def get_pointer():
        return self.t.data_ptr()


def bresenham_2d(pts_source, pts_target, map_size):
    """
    2D Bresenham line algorithm implemented by PyCUDA for GPU acceleration.
    """
    if isinstance(map_size, list):
        assert len(map_size) == 2 and map_size[0] == map_size[1], "map_size must be a square"
        map_size = map_size[0]

    # Keep data on GPU if already there
    device = pts_source.device
    source_pts = pts_source.int().contiguous().to(device)
    target_pts = pts_target.int().contiguous().to(device)
    num_rays = target_pts.shape[0]
    
    # Optimize max_pts_per_ray calculation based on manhattan distance
    # max_pts_per_ray = min(map_size * 2, 
    #     int(1.2 * torch.max(torch.abs(target_pts - source_pts.expand_as(target_pts)).sum(dim=1))))
    max_pts_per_ray = map_size * 2

    # Allocate output memory directly on GPU
    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 2), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros(num_rays, dtype=torch.int32, device=device)

    kernel_code = """
    __device__ __forceinline__ void bresenham_line(
        const int x0, const int y0,
        const int x1, const int y1,
        int *__restrict__ trajectory,
        int *__restrict__ length,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int dx = abs(x1 - x0);
        const int dy = abs(y1 - y0);
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        
        int x = x0;
        int y = y0;
        int err = dx - dy;
        int idx = 0;
        
        // Pre-compute bounds check for first point
        const bool initial_valid = (x >= 0 && x < map_size && y >= 0 && y < map_size);
        if (initial_valid) {
            trajectory[0] = x;
            trajectory[1] = y;
            idx = 1;
        }
        
        #pragma unroll 4
        while (idx < max_pts_per_ray) {
            if (x == x1 && y == y1) break;
            
            const int e2 = 2 * err;
            
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
            
            if (x >= 0 && x < map_size && y >= 0 && y < map_size) {
                trajectory[idx * 2] = x;
                trajectory[idx * 2 + 1] = y;
                idx++;
            }
        }
        
        *length = idx;
    }

    __global__ void ray_casting_kernel(
        const int *__restrict__ source_pts,
        const int *__restrict__ target_pts,
        int *__restrict__ trajectory_pts,
        int *__restrict__ trajectory_lengths,
        const int num_rays,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) return;
        
        const int src_x = source_pts[0];
        const int src_y = source_pts[1];
        const int tgt_x = target_pts[ray_idx * 2];
        const int tgt_y = target_pts[ray_idx * 2 + 1];
        
        bresenham_line(
            src_x, src_y,
            tgt_x, tgt_y,
            &trajectory_pts[ray_idx * max_pts_per_ray * 2],
            &trajectory_lengths[ray_idx],
            map_size,
            max_pts_per_ray
        );
    }
    """
    
    # Compile kernel with optimization flags
    mod = SourceModule(kernel_code, options=['-O3'])
    ray_casting_kernel = mod.get_function("ray_casting_kernel")
    
    # Configure kernel launch parameters
    block_size = 256
    grid_size = (num_rays + block_size - 1) // block_size
    
    # Launch kernel with streamed execution
    stream = cuda.Stream()
    ray_casting_kernel(
        source_pts,
        target_pts,
        trajectory_pts,
        trajectory_lengths,
        np.int32(num_rays),
        np.int32(map_size),
        np.int32(max_pts_per_ray),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        stream=stream
    )
    
    # Process results efficiently using GPU operations
    mask = torch.arange(max_pts_per_ray, device=device)[None, :] < trajectory_lengths[:, None]
    mask = mask.unsqueeze(-1).expand(-1, -1, 2)
    results = trajectory_pts[mask].view(-1, 2)

    return results.to(torch.long)


def extract_ego_maps(global_maps, cell_sizes, poses_idx, ego_cm=10):
    """
    Extract ego maps (cropped and resampled) from global maps.
    
    In this updated version, ego_cm represents the desired physical size (in cm)
    of each cell in the output ego map. Since the ego map has shape [H, W],
    the overall physical extent of the extracted region is [H * ego_cm, W * ego_cm] cm.
    
    This vectorized implementation removes the explicit for-loop.
    
    Args:
        global_maps (torch.Tensor): Global maps of shape [N, H, W].
        cell_sizes (torch.Tensor): Tensor of shape [N, 2] where each row is [cell_height_cm, cell_width_cm]
                                for the global map.
        poses_idx (torch.Tensor): Tensor of shape [N, 2] with the center pose (row, col) in pixel coordinates.
        ego_cm (float): Desired physical size (cm) for each cell in the ego map.
        
    Returns:
        torch.Tensor: Ego maps of shape [N, H, W] extracted from each environment.
    """
    N, H, W = global_maps.shape
    device = global_maps.device

    # Total physical extent (in cm) of the extracted patch
    H_cm = H * ego_cm  # total patch height in cm
    W_cm = W * ego_cm  # total patch width in cm

    # Compute patch size in pixels for each environment using cell_sizes [cm/pixel]
    patch_h_pixels = H_cm / cell_sizes[:, 0]  # shape: [N]
    patch_w_pixels = W_cm / cell_sizes[:, 1]  # shape: [N]
    half_patch_h = patch_h_pixels / 2.0         # shape: [N]
    half_patch_w = patch_w_pixels / 2.0         # shape: [N]

    # Create interpolation parameters t ranging from 0 to 1 for H and W steps.
    # These will be used to linearly interpolate between the patch boundaries.
    t_y = torch.linspace(0, 1, steps=H, device=device).unsqueeze(0)  # shape: [1, H]
    t_x = torch.linspace(0, 1, steps=W, device=device).unsqueeze(0)  # shape: [1, W]

    # Compute the starting coordinates (upper-most/left-most) for the patch in each environment.
    start_y = poses_idx[:, 0].unsqueeze(1) - half_patch_h.unsqueeze(1)  # shape: [N, 1]
    start_x = poses_idx[:, 1].unsqueeze(1) - half_patch_w.unsqueeze(1)  # shape: [N, 1]

    # Compute the full y and x coordinates for each environment by interpolating between the patch boundaries.
    # For each environment: coordinate = start + (patch_pixels * t)
    y_coords = start_y + patch_h_pixels.unsqueeze(1) * t_y      # shape: [N, H]
    x_coords = start_x + patch_w_pixels.unsqueeze(1) * t_x      # shape: [N, W]

    # Create meshgrids for each environment without an explicit loop.
    # grid_y: [N, H, W] and grid_x: [N, H, W]
    grid_y = y_coords.unsqueeze(2).expand(-1, -1, W)
    grid_x = x_coords.unsqueeze(1).expand(-1, H, -1)

    # Normalize these pixel coordinates to [-1, 1] (the range expected by grid_sample).
    # Here, (H - 1) and (W - 1) are used based on the global map dimensions.
    norm_y = (grid_y / (H - 1)) * 2 - 1
    norm_x = (grid_x / (W - 1)) * 2 - 1

    # Combine normalized x and y (note grid_sample expects (x, y) order).
    batch_grid = torch.stack([norm_x, norm_y], dim=-1)  # shape: [N, H, W, 2]

    # Sample the ego maps using grid_sample with nearest neighbor interpolation.
    ego_maps = F.grid_sample(
        input=global_maps.unsqueeze(1),  # [N, 1, H, W]
        grid=batch_grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )

    return ego_maps.squeeze(1)  # [N, H, W]


def compute_frontier_map(ego_prob_maps, frontier_kernel):
    """
    Args:
        ego_prob_maps: [num_env, H, W], cell values among {-1, 0, 1}
    Returns:
        frontier_map: [num_env, H, W], cell values among {0, 1}
    """
    # ego_prob_maps_denoised = ego_prob_maps.clone()
    ego_prob_maps_denoised = F.max_pool2d(ego_prob_maps.unsqueeze(1), kernel_size=3, stride=1, padding=1)  # [num_env, 1, H, W]

    # Create masks for -1 and 0 pixels
    mask_neg1 = (ego_prob_maps_denoised == -1).float()
    mask_unknown = (ego_prob_maps_denoised == 0).float()

    # Apply convolution to mask_0 to count number of 0 neighbors for each pixel
    unknown_neighbors = F.conv2d(mask_unknown, frontier_kernel, padding=1)

    # A frontier boundary pixel is a -1 pixel with at least one 0 neighbor
    frontier = (mask_neg1 == 1) & (unknown_neighbors >= 1)

    return frontier.squeeze(1)  # [num_env, H, W]


def create_e2w_from_poses(poses: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    2D version for action space (x, y, yaw). Compute the transformation matrices from egocentric to world coordinates.

    Args:
        poses (torch.Tensor): Tensor of shape [num_env, 6] containing 
                                [x, y, z, roll, pitch, yaw] for each environment.
        device (torch.device): The device on which to perform computations.

    Returns:
        torch.Tensor: Transformation matrices of shape [num_env, 3, 3].
    """
    num_env = poses.shape[0]

    # Extract relevant components for 2D transformation
    x = poses[:, 0]        # [num_env]
    y = poses[:, 1]        # [num_env]
    yaw = poses[:, 5]      # [num_env]

    # Compute cosine and sine of yaw angles
    cos_yaw = torch.cos(yaw).unsqueeze(1)  # [num_env, 1]
    sin_yaw = torch.sin(yaw).unsqueeze(1)  # [num_env, 1]

    # Prepare the rotation and translation parts of the matrix
    # Rotation matrix components
    rotation_00 = cos_yaw                     # [num_env, 1]
    rotation_01 = -sin_yaw                    # [num_env, 1]
    rotation_10 = sin_yaw                     # [num_env, 1]
    rotation_11 = cos_yaw                     # [num_env, 1]

    # Translation components
    translation_x = x.unsqueeze(1)            # [num_env, 1]
    translation_y = y.unsqueeze(1)            # [num_env, 1]

    # Construct the 2D affine transformation matrix in homogeneous coordinates
    # Each matrix looks like:
    # [ cos(yaw)  -sin(yaw)   x ]
    # [ sin(yaw)   cos(yaw)   y ]
    # [    0          0        1 ]
    e2w_transformation_matrix = torch.cat([
        torch.cat([rotation_00, rotation_01, translation_x], dim=1).unsqueeze(1),  # First row
        torch.cat([rotation_10, rotation_11, translation_y], dim=1).unsqueeze(1),  # Second row
        torch.tensor([0., 0., 1.], device=device).expand(num_env, 1, 3)            # Third row
    ], dim=1)  # Shape: [num_env, 3, 3]

    return e2w_transformation_matrix


def scanned_pts_to_2d_idx(pts_target, range_gt_scenes, voxel_size_scenes, motion_height=1.0, map_size=256, return_mask=False):
    """
    Args:
        pts_target: [num_env, num_pts, 3], list of target points by back-projection

        range_gt_scenes: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size_scenes: [num_env, 3]

    Return:
        pts_target_idxs: list of (num_valid_pts_idx, 2), torch.long, used for indexing
    """
    num_env = pts_target.shape[0]

    motion_height_idx = ((motion_height - range_gt_scenes[:, 5]) / voxel_size_scenes[:, 2]).long()    # [num_env]
    xyz_max_voxel = range_gt_scenes[:, [0,2,4]] + 0.5 * voxel_size_scenes
    xyz_min_voxel = range_gt_scenes[:, [1,3,5]] - 0.5 * voxel_size_scenes

    # [num_env, num_pts, 3], convert to indices
    pts_target_idx = torch.floor(
        (pts_target - xyz_min_voxel.unsqueeze(1)) / voxel_size_scenes.unsqueeze(1)
    ).long()

    # [num_env, num_pts, 3], bounds checking masks
    bound_mask = (xyz_max_voxel.unsqueeze(1) > pts_target) & (pts_target > xyz_min_voxel.unsqueeze(1))
    bound_mask = bound_mask.all(dim=-1)  # [num_env, num_pts]
    height_mask = pts_target_idx[..., 2] == motion_height_idx.unsqueeze(1)    # [num_env, num_pts], height mask
    final_mask = bound_mask & height_mask

    pts_target_idxs = []
    for env_idx in range(num_env):
        # Get valid points for this environment
        valid_pts = pts_target_idx[env_idx][final_mask[env_idx]]

        if valid_pts.shape[0] == 0:
            pts_target_idxs.append([])
            continue

        # Unique and clip
        valid_pts = torch.unique(valid_pts, dim=0)
        valid_pts = torch.clip(valid_pts, min=0, max=map_size-1)
        pts_target_idxs.append(valid_pts[:, :2])    # [num_valid_pts, 2]
    if return_mask:
        return pts_target_idxs, final_mask
    return pts_target_idxs


def scanned_pts_to_2d_idx_vis(pts_target, clr_target, range_gt_scenes, voxel_size_scenes, motion_height=1, scale=1):
    """
    Args:
        pts_target: [num_env, num_pts, 3], list of target points by back-projection

        range_gt: [num_obj, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size: [num_obj, 3]

    Return:
        pts_target_idxs: list of (num_valid_pts_idx, 3)
    """

    num_env = pts_target.shape[0]

    motion_height_idx = ((motion_height - range_gt_scenes[:, 5]) / voxel_size_scenes[:, 2]).long()    # [num_env]
    xyz_max_voxel = range_gt_scenes[:, [0,2,4]] + 0.5 * voxel_size_scenes
    xyz_min_voxel = range_gt_scenes[:, [1,3,5]] - 0.5 * voxel_size_scenes

    # [num_env, num_pts, 3], convert to indices
    pts_target_idx = torch.floor(
        (pts_target - xyz_min_voxel.unsqueeze(1)) / voxel_size_scenes.unsqueeze(1)
    ).long()

    # [num_env, num_pts, 3], bounds checking masks
    bound_mask = (xyz_max_voxel.unsqueeze(1) > pts_target) & (pts_target > xyz_min_voxel.unsqueeze(1))
    bound_mask = bound_mask.all(dim=-1)  # [num_env, num_pts]
    height_mask = pts_target_idx[..., 2] == motion_height_idx.unsqueeze(1)    # [num_env, num_pts], height mask
    final_mask = bound_mask & height_mask

    pts_target_idxs = []
    clr_targets = []
    for env_idx in range(num_env):
        # Get valid points for this environment
        valid_pts = pts_target_idx[env_idx][final_mask[env_idx]]
        valid_clr = clr_target[env_idx][final_mask[env_idx]]

        if valid_pts.shape[0] == 0:
            pts_target_idxs.append([])
            continue

        # Unique and clip
        valid_pts, unique_idx = torch.unique(valid_pts, dim=0, return_inverse=True)
        valid_pts = torch.clip(valid_pts, min=0, max=256//scale-1)
        pts_target_idxs.append(valid_pts)


        unique_indices = torch.unique(unique_idx, return_inverse=True)[1]
        mask = torch.zeros_like(unique_idx, dtype=torch.bool)
        mask[unique_indices] = True
        
        # Use the mask to get the indices of the first occurrence of each unique element
        indices_of_unique_elements = torch.where(mask)[0]
        valid_clr = valid_clr[indices_of_unique_elements]
        clr_targets.append(valid_clr)

    return pts_target_idxs, clr_targets


def pose_coord_to_2d_idx(poses, range_gt_scenes, voxel_size_scenes, map_size=256):
    """
    Args:
        poses: [num_step, 2] or [num_env, num_step, 2], x-y pose

        range_gt_scenes: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size_scenes: [num_env, 3]

    Return:
        poses_idx: [num_env, 2] or [num_env, num_step, 2]
    """
    x_min = range_gt_scenes[:, 1]   # [num_env]
    y_min = range_gt_scenes[:, 3]   # [num_env]
    voxel_sizes_xy = voxel_size_scenes[:, :2]  # [num_env, 2]
    xy_min_voxel = torch.stack([x_min, y_min], dim=-1) - 0.5 * voxel_sizes_xy  # [num_env, 2]

    # Handle different input shapes
    if poses.dim() == 2:
        # [num_env, 2]
        poses_idx = ((poses - xy_min_voxel) / voxel_sizes_xy).floor().long()
    elif poses.dim() == 3:
        # [num_env, num_step, 2]
        poses_idx = ((poses - xy_min_voxel.unsqueeze(1)) / voxel_sizes_xy.unsqueeze(1)).floor().long()
    else:
        raise ValueError(f"Invalid poses shape: {poses.shape}")
    
    poses_idx = torch.clip(poses_idx, min=0, max=map_size-1)
    return poses_idx


def pose_coord_to_2d_idx_vis(poses, range_gt_scenes, voxel_size_scenes):
    """
    Args:
        poses: [num_step, 2], x-y pose

        range_gt_scenes: [6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt for the specific scene
        voxel_size_scenes: [3]

    Return:
        poses_idx: [num_step, 2]
    """
    x_min = range_gt_scenes[1]
    y_min = range_gt_scenes[3]
    voxel_sizes_xy = voxel_size_scenes[:2]  # [2]
    xy_min_voxel = torch.stack([x_min, y_min], dim=-1) - 0.5 * voxel_sizes_xy  # [2]

    # Handle different input shapes using a single computation path
    if poses.dim() == 2:
        # [num_env, 2]
        poses_idx = ((poses - xy_min_voxel.unsqueeze(0)) / voxel_sizes_xy.unsqueeze(0)).floor().long()
    else:
        raise ValueError(f"Invalid poses shape: {poses.shape}")
    
    return poses_idx.clamp_(min=0, max=255)


def pts_coord_to_2d_idx(pts_target, range_gt, voxel_size, env_to_scene):
    """
    Args:
        pts_target: [num_env, num_pts, 2], list of target points (2D) by back-projection

        range_gt: [num_obj, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size: [num_obj, 3]

        env_to_scene: [num_env] tensor, index of scene or object

    Return:
        pts_target_idxs: list of (num_valid_pts_idx, 3)
    """

    x_max, y_max = range_gt[env_to_scene, 0], range_gt[env_to_scene, 2] # [num_env]
    x_min, y_min = range_gt[env_to_scene, 1], range_gt[env_to_scene, 3] # [num_env]

    x_max_voxel = x_max + 1 / 2 * voxel_size[env_to_scene, 0] # [num_env]
    y_max_voxel = y_max + 1 / 2 * voxel_size[env_to_scene, 1]
    x_min_voxel = x_min - 1 / 2 * voxel_size[env_to_scene, 0]
    y_min_voxel = y_min - 1 / 2 * voxel_size[env_to_scene, 1] 
    xy_min_voxel = torch.stack([x_min_voxel, y_min_voxel], dim=-1) # [num_env, 2]

    pts_target_idxs = []
    for env_idx, pts in enumerate(pts_target):
        if len(pts.shape) == 1: # for pose_idx_2D
            pts = pts.unsqueeze(0)
        pts = pts[:, :2]

        # [H*W], bool tensor
        valid_pts = (x_max_voxel[env_idx] > pts[:, 0]) * (pts[:, 0] > x_min_voxel[env_idx]) * \
                    (y_max_voxel[env_idx] > pts[:, 1]) * (pts[:, 1] > y_min_voxel[env_idx])

        pts = pts[valid_pts]  # [num_valid_pts, 3], filter valid points

        pts_target_idx = torch.floor((pts - xy_min_voxel[env_idx]) / voxel_size[env_to_scene][env_idx, :2]).long()    # [num_tar_pts, 2]
        if pts_target_idx.shape[0] == 0:
            continue
        pts_target_idx = torch.unique(pts_target_idx, dim=0)   # [num_valid_tar_pts, 2], speed up a lot
        pts_target_idx = torch.clip(pts_target_idx, min=0, max=255)
        pts_target_idxs.append(pts_target_idx)

    return pts_target_idxs


def discretize_prob_map(grid_prob, threshold_occu=0.5, threshold_free=0.0, return_tri_cls_only=False):
    """
    Args:
        grid_prob: [num_env, X, Y, Z]

    Return:
        grid_occupancy: [num_env, X, Y, Z], voxel value among {0/1}. 0: free/unknown, 1: occupied
        grid_tri_cls: [num_env, X, Y, Z], voxel value among {-1/0/1}. -1: free, 0: unknown, 1: occupied
    """
    grid_occupancy = (grid_prob > threshold_occu).to(torch.float32)
    grid_free = (grid_prob < threshold_free).to(torch.float32)

    grid_tri_cls = grid_occupancy - grid_free   # element value: {-1, 0, 1}
    if return_tri_cls_only:
        return grid_tri_cls
    else:
        return grid_occupancy, grid_tri_cls


def visualize_grid(occ_map, path, start, end):
    """
    Visualizes the occupancy grid with trajectories by pathfinding.
    """
    if isinstance(occ_map, torch.Tensor):
        occ_map = occ_map.cpu().numpy()

    grid = np.array(occ_map)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='gray_r')

    # Plot start and end points
    plt.scatter(start[1], start[0], c='red', marker='*', s=130, label='Start')
    plt.scatter(end[1], end[0], c='orange', marker='X', s=100, label='End')

    # Plot path if exists
    if path:
        path_x = [cell.x for cell in path]
        path_y = [cell.y for cell in path]
        plt.plot(path_x, path_y, marker='o', color='blue', label='Path', alpha=0.7, markersize=5)

    plt.legend()
    plt.title('A* Pathfinding Visualization')
    plt.gca()
    plt.show()


def getURDFParameter(urdf_path, parameter_name: str):
    """Reads a parameter from a drone's URDF file.

    This method is nothing more than a custom XML parser for the .urdf
    files in folder `assets/`.

    Parameters
    ----------
    parameter_name : str
        The name of the parameter to read.

    Returns
    -------
    float
        The value of the parameter.

    """
    #### Get the XML tree of the drone model to control ########
    path = urdf_path
    URDF_TREE = etxml.parse(path).getroot()
    #### Find and return the desired parameter #################
    if parameter_name == 'm':
        return float(URDF_TREE[1][0][1].attrib['value'])
    elif parameter_name in ['ixx', 'iyy', 'izz']:
        return float(URDF_TREE[1][0][2].attrib[parameter_name])
    elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                            'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
        return float(URDF_TREE[0].attrib[parameter_name])
    elif parameter_name in ['length', 'radius']:
        return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
    elif parameter_name == 'collision_z_offset':
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        return COLLISION_SHAPE_OFFSETS[2]

