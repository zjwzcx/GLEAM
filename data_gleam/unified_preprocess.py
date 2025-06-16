import os
import torch
import numpy as np
import open3d as o3d
import h5py
import time
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from legged_gym import OPEN_ROBOT_ROOT_DIR


# ===== Common Utility Functions ==============================================

@torch.jit.script
def points2voxel_grid(points: torch.Tensor, grid_size: torch.Tensor, pts_range: torch.Tensor) -> torch.Tensor:
    """Optimized voxelization function"""
    # Extract range boundaries
    x_max, x_min = pts_range[0].item(), pts_range[1].item()
    y_max, y_min = pts_range[2].item(), pts_range[3].item()
    z_max, z_min = pts_range[4].item(), pts_range[5].item()

    # Create coordinate grids
    xs = torch.linspace(x_min, x_max, steps=int(grid_size[0].item()), device=points.device)
    ys = torch.linspace(y_min, y_max, steps=int(grid_size[1].item()), device=points.device)
    zs = torch.linspace(z_min, z_max, steps=int(grid_size[2].item()), device=points.device)

    # Calculate voxel sizes
    voxel_size = torch.stack([xs[1] - xs[0], 
                             ys[1] - ys[0], 
                             zs[1] - zs[0]]).to(torch.float32)

    # Calculate voxel grid minima
    x_min_voxel = torch.tensor(x_min, dtype=torch.float32, device=points.device) - 0.5 * voxel_size[0]
    y_min_voxel = torch.tensor(y_min, dtype=torch.float32, device=points.device) - 0.5 * voxel_size[1]
    z_min_voxel = torch.tensor(z_min, dtype=torch.float32, device=points.device) - 0.5 * voxel_size[2]

    # Create voxel grid coordinates
    xs, ys, zs = torch.meshgrid(xs, ys, zs, indexing='ij')
    voxels = torch.stack([xs, ys, zs], dim=-1)
    # Add occupancy channel
    voxels = torch.cat((voxels, torch.zeros(voxels.shape[:-1], device=points.device).unsqueeze(-1)), dim=-1)

    # Calculate point coordinates in voxel grid
    pts_coord = torch.stack([
        torch.floor((points[:, 0] - x_min_voxel) / voxel_size[0]),
        torch.floor((points[:, 1] - y_min_voxel) / voxel_size[1]),
        torch.floor((points[:, 2] - z_min_voxel) / voxel_size[2])
    ], dim=-1).long()

    # Mark occupied voxels
    voxels.index_put_((pts_coord[:, 0], pts_coord[:, 1], pts_coord[:, 2], torch.tensor(3)), 
                      torch.ones(pts_coord.shape[0], device=points.device))

    # Special handling for floor points
    floor_mask = (pts_coord[:, 2] == pts_coord[:, 2].min()) & (pts_coord[:, 2] == (pts_coord[:, 2].min()+1))
    pts_coord_floor = pts_coord[floor_mask]
    voxels.index_put_((pts_coord_floor[:, 0], pts_coord_floor[:, 1], pts_coord_floor[:, 2], torch.tensor(3)), 
                      torch.zeros(pts_coord_floor.shape[0], device=points.device))

    return voxels

@torch.jit.script
def voxels_occupancy_surface(voxels: torch.Tensor, only_return_pcd: bool = True) -> torch.Tensor:
    """Extract voxel surface points"""
    # Initialize output tensor
    voxels_occupied = torch.zeros_like(voxels)
    voxels_occupied[..., :3] = voxels[..., :3]
    
    # Create occupancy mask
    occupied = voxels[..., 3] > 0
    # Pad occupancy for neighbor checks
    padded = torch.nn.functional.pad(occupied, (1, 1, 1, 1, 1, 1), mode='constant', value=1.)
    
    # Extract neighbor information
    x_minus = padded[:-2, 1:-1, 1:-1]
    x_plus = padded[2:, 1:-1, 1:-1]
    y_minus = padded[1:-1, :-2, 1:-1]
    y_plus = padded[1:-1, 2:, 1:-1]
    z_minus = padded[1:-1, 1:-1, :-2]
    z_plus = padded[1:-1, 1:-1, 2:]
    
    # Identify surface voxels
    neighbor_product = x_minus & x_plus & y_minus & y_plus & z_minus & z_plus
    surface_mask = occupied & (~neighbor_product)
    
    # Handle boundary voxels
    boundary_mask = (
        (torch.arange(voxels.shape[0], device=voxels.device) == voxels.shape[0]-1).view(-1, 1, 1) |
        (torch.arange(voxels.shape[1], device=voxels.device) == voxels.shape[1]-1).view(1, -1, 1) |
        (torch.arange(voxels.shape[2], device=voxels.device) == voxels.shape[2]-1).view(1, 1, -1)
    )
    surface_mask = surface_mask | (occupied & boundary_mask)
    
    # Extract surface points
    surface_indices = torch.nonzero(surface_mask)
    surface_occupied_pts = voxels[surface_indices[:, 0], surface_indices[:, 1], surface_indices[:, 2], :3]
    
    # Update occupancy information
    voxels_occupied[..., 3] = torch.where(surface_mask, voxels[..., 3], torch.zeros_like(voxels[..., 3]))
    
    if only_return_pcd:
        return surface_occupied_pts
    else:
        return surface_occupied_pts, voxels_occupied

def pcd_maxmin(pcd: torch.Tensor, print_range: bool = True) -> list:
    """Calculate point cloud boundaries"""
    mins, maxs = pcd.min(dim=0)[0], pcd.max(dim=0)[0]
    if print_range:
        print(f"X: {mins[0].item():.4f}~{maxs[0].item():.4f}, "
              f"Y: {mins[1].item():.4f}~{maxs[1].item():.4f}, "
              f"Z: {mins[2].item():.4f}~{maxs[2].item():.4f}")
    return [maxs[0].item(), mins[0].item(),
            maxs[1].item(), mins[1].item(),
            maxs[2].item(), mins[2].item()]

def get_mesh_number(filename: str) -> int:
    """Extract scene number for sorting"""
    return int(filename.replace('.ply', '').replace('scene_', ''))

def tensor_to_dict(tensor: torch.Tensor) -> dict:
    """Convert voxel tensor to dictionary format"""
    num_scene, _, _, _, C = tensor.shape
    assert C == 4, "Tensor must have 4 channels (x, y, z, occupancy)"
    result = {}

    for scene_idx in range(num_scene):
        scene_data = tensor[scene_idx]
        # Find indices of occupied voxels
        occupied_voxels = torch.nonzero(scene_data[..., 3] == 1)
        indices = occupied_voxels
        coordinates = scene_data[occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2], :3]
        # Combine indices and coordinates
        idx_coord = torch.cat((indices, coordinates), dim=-1)
        result[scene_idx] = idx_coord

    return result

# ===== Main Processing Pipeline =============================================

def preprocess_meshes(dataset_name: str):
    """Step 1: Mesh preprocessing - centering and scaling ply meshes"""
    print(f"\n=== Step 1: Preprocess meshes [{dataset_name}] ===")
    # Get scene mesh files
    scene_meshes = os.listdir(f"data_gleam/customized_data/objects/{dataset_name}_ply_raw")
    scene_meshes = [f for f in scene_meshes if f.endswith(".ply")]
    scene_meshes.sort(key=get_mesh_number)

    # Create output directory
    save_dir = f"data_gleam/customized_data/objects/{dataset_name}_ply_center"
    os.makedirs(save_dir, exist_ok=True)
    scale = 100.

    for scene_idx, scene_name in enumerate(scene_meshes):
        print(f"Processing scene: {scene_name}")
        mesh_path = f"data_gleam/customized_data/objects/{dataset_name}_ply_raw/{scene_name}"
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if mesh.is_empty():
            print(f"  Skip empty mesh: {scene_name}")
            continue
            
        mesh_vertices = np.asarray(mesh.vertices)
        pc_range = pcd_maxmin(torch.tensor(mesh_vertices), True)
        
        # Translate mesh to center
        mesh_translate = np.array([
            [-(pc_range[0] + pc_range[1]) / 2],
            [-(pc_range[2] + pc_range[3]) / 2],
            [-pc_range[5]]
        ])
        mesh.translate(mesh_translate)
        
        # Scale handling
        if pc_range[4] - pc_range[5] < 100.:
            print("  Skip scaling")
        else:
            mesh_vertices = np.asarray(mesh.vertices) / scale
            print(f"  Scaled by {scale}x")
            mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        
        # Save processed mesh
        save_path = os.path.join(save_dir, f"scene_{scene_idx}.ply")
        o3d.io.write_triangle_mesh(save_path, mesh)
        print(f"  Saved to: {save_path}")

def voxelize_scenes(dataset_name: str, grid_reso: int = 128):
    """Step 2: Voxelize scenes"""
    print(f"\n=== Step 2: Voxelize scenes [{dataset_name}], resolution {grid_reso} ===")
    base_path = os.path.join(OPEN_ROBOT_ROOT_DIR, "data_gleam")
    input_dir = f"objects/{dataset_name}_ply_center"
    scene_names = [f[:-4] for f in os.listdir(os.path.join(base_path, input_dir)) 
                  if f.endswith(".ply")]
    scene_names.sort(key=get_mesh_number)

    # Create output directories
    out_dir = os.path.join(base_path, f"gt/gt_{dataset_name}")
    per_scene_dir = os.path.join(out_dir, "gt_per_scene")
    os.makedirs(per_scene_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    grid_size = torch.tensor([grid_reso, grid_reso, grid_reso])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for scene_idx, scene_name in enumerate(scene_names):
        start_time = time.time()
        print(f"Processing scene {scene_idx}/{len(scene_names)}: {scene_name}")
        
        # Load mesh and sample point cloud
        mesh_path = os.path.join(base_path, input_dir, f"scene_{scene_idx}.ply")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = mesh.sample_points_poisson_disk(number_of_points=1000000)
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
        pts_range = torch.tensor(pcd_maxmin(points, False), device=device)
        
        # Voxelization
        voxels = points2voxel_grid(points, grid_size.to(device), pts_range)
        pts_occupied_surface, grid_occupied_surface = voxels_occupancy_surface(
            voxels, only_return_pcd=False)
        
        # Save results
        pcd_save_path = os.path.join(per_scene_dir, f"{scene_name}_{grid_reso}_pc.pcd")
        pcd_surface = o3d.geometry.PointCloud()
        pcd_surface.points = o3d.utility.Vector3dVector(pts_occupied_surface.cpu().numpy())
        o3d.io.write_point_cloud(pcd_save_path, pcd_surface)
        
        grid_save_path = os.path.join(per_scene_dir, f"{scene_name}_{grid_reso}_grid_gt.pt")
        torch.save(grid_occupied_surface.unsqueeze(0).cpu(), grid_save_path)
        
        print(f"  Completed! Time: {time.time()-start_time:.2f}s")
        print(f"  Point cloud saved to: {pcd_save_path}")
        print(f"  Voxel grid saved to: {grid_save_path}")

def merge_voxel_data(dataset_name: str, grid_reso: int = 128):
    """Step 3: Merge voxel data"""
    print(f"\n=== Step 3: Merge voxel data [{dataset_name}], resolution {grid_reso} ===")
    load_path = os.path.join(OPEN_ROBOT_ROOT_DIR, 
                            f"data_gleam/customized_data/gt/gt_{dataset_name}/gt_per_scene/")
    save_path = os.path.join(OPEN_ROBOT_ROOT_DIR, 
                            f"data_gleam/customized_data/gt/gt_{dataset_name}/")
    
    scene_files = [f for f in os.listdir(load_path) if f.endswith(".pt") and f"_{grid_reso}_" in f]
    scene_files.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(scene_files)} voxel files")
    grids_gt, voxel_size_gt, num_valid_voxel_gt, range_gt = [], [], [], []
    
    for scene_file in scene_files:
        scene_idx = int(scene_file.split('_')[1])
        print(f"Loading scene {scene_idx}: {scene_file}")
        
        grid_gt = torch.load(os.path.join(load_path, scene_file))
        assert len(grid_gt.shape) == 5 and grid_gt.shape[-1] == 4 and grid_gt.shape[0] == 1
        
        # Calculate voxel sizes
        voxel_size = [
            (grid_gt[0, 1, 0, 0, 0] - grid_gt[0, 0, 0, 0, 0]).item(),
            (grid_gt[0, 0, 1, 0, 1] - grid_gt[0, 0, 0, 0, 1]).item(),
            (grid_gt[0, 0, 0, 1, 2] - grid_gt[0, 0, 0, 0, 2]).item()
        ]
        
        grids_gt.append(grid_gt)
        voxel_size_gt.append(voxel_size)
        num_valid_voxel_gt.append(grid_gt[..., 3].sum())
        range_gt.append(pcd_maxmin(grid_gt, False))
    
    # Merge all scene data
    grid_gt = torch.cat(grids_gt, dim=0)
    voxel_size_gt = torch.tensor(voxel_size_gt)
    num_valid_voxel_gt = torch.stack(num_valid_voxel_gt)
    range_gt = torch.tensor(range_gt)
    grid_gt_occ = grid_gt.clone()[..., 3]
    
    # Save merged data
    base_name = f"{dataset_name}_{grid_reso}"
    torch.save(grid_gt.cpu(), os.path.join(save_path, f"{base_name}_4_grid_gt.pt"))
    torch.save(grid_gt_occ.cpu(), os.path.join(save_path, f"{base_name}_grid_gt.pt"))
    torch.save(voxel_size_gt.cpu(), os.path.join(save_path, f"{base_name}_voxel_size_gt.pt"))
    torch.save(num_valid_voxel_gt.cpu(), os.path.join(save_path, f"{base_name}_num_valid_voxel_gt.pt"))
    torch.save(range_gt.cpu(), os.path.join(save_path, f"{base_name}_range_gt.pt"))

    print(f"Merging complete! Total scenes: {len(scene_files)}")
    print(f"Saved files to: {save_path}")

def generate_2d_maps(dataset_name: str, grid_reso: int = 128, height: float = 1.5):
    """Step 4: Generate 2D maps"""
    print(f"\n=== Step 4: Generate 2D maps [{dataset_name}], height {height}m ===")
    root_path = os.path.join(OPEN_ROBOT_ROOT_DIR, f"data_gleam/customized_data/gt/gt_{dataset_name}")
    grid_gt = torch.load(os.path.join(root_path, f"{dataset_name}_{grid_reso}_grid_gt.pt")).squeeze(1)
    range_gt = torch.load(os.path.join(root_path, f"{dataset_name}_{grid_reso}_range_gt.pt"))
    voxel_size_gt = torch.load(os.path.join(root_path, f"{dataset_name}_{grid_reso}_voxel_size_gt.pt"))
    
    # Generate initial map
    print("Generating initial map...")
    occ_maps_sum = grid_gt[..., -20:].sum(dim=-1)
    occ_maps_sum[occ_maps_sum > 0] = 255.
    occ_maps_sum[occ_maps_sum <= 0] = 0.
    
    # Calculate height indices
    height_idx = ((height - range_gt[:, 5]) / voxel_size_gt[:, 2]).long()
    min_idx = torch.clamp(height_idx - 5, min=0)
    max_idx = torch.clamp(height_idx + 5, max=grid_gt.shape[-1])
    
    # Create plane occupancy map
    occ_maps_plane = torch.zeros(grid_gt.shape[:-1])
    for i in range(grid_gt.shape[0]):
        occ_maps_plane[i] = grid_gt[i, :, :, min_idx[i]:max_idx[i]].sum(dim=-1)
    occ_maps_plane[occ_maps_plane > 0] = 255.
    occ_maps_plane[occ_maps_plane <= 0] = 0.
    
    # Create initial map
    init_maps = occ_maps_sum - occ_maps_plane
    pooled_maps = 255. - init_maps
    pooled_maps = F.max_pool2d(pooled_maps, kernel_size=9, stride=1, padding=4)
    init_maps = 255. - pooled_maps
    
    # Save initial map
    init_save_path = os.path.join(root_path, f"{dataset_name}_{grid_reso}_init_map_1d5.pt")
    torch.save(init_maps, init_save_path)
    print(f"Initial map saved to: {init_save_path}")
    
    # Generate occupancy map
    print("Generating occupancy map...")
    height_idx = ((height - range_gt[:, 5]) / voxel_size_gt[:, 2]).long()
    occupancy_maps_gt = torch.zeros(grid_gt.shape[0], grid_gt.shape[1], grid_gt.shape[2])
    
    for env_idx in range(grid_gt.shape[0]):
        occupancy_maps_gt[env_idx] = grid_gt[env_idx, :, :, height_idx[env_idx]-1:height_idx[env_idx]+1].sum(dim=-1)
    
    # Binarize occupancy map
    occupancy_maps_gt[occupancy_maps_gt > 0] = 255.
    occupancy_maps_gt[occupancy_maps_gt <= 0] = 0.
    
    # Optimize occupancy map (remove internal points)
    for env_idx in range(occupancy_maps_gt.shape[0]):
        occupancy_map = occupancy_maps_gt[env_idx]
        # Calculate neighbor shifts
        up = torch.roll(occupancy_map, shifts=1, dims=0)
        down = torch.roll(occupancy_map, shifts=-1, dims=0)
        left = torch.roll(occupancy_map, shifts=1, dims=1)
        right = torch.roll(occupancy_map, shifts=-1, dims=1)
        up_left = torch.roll(occupancy_map, shifts=(1, 1), dims=(0, 1))
        up_right = torch.roll(occupancy_map, shifts=(1, -1), dims=(0, 1))
        down_left = torch.roll(occupancy_map, shifts=(-1, 1), dims=(0, 1))
        down_right = torch.roll(occupancy_map, shifts=(-1, -1), dims=(0, 1))
        
        # Create masks for boundary detection
        four_neighbors_mask = ((up == 255.) & (down == 255.) & (left == 255.) & (right == 255.))
        internal_mask = (four_neighbors_mask & (occupancy_map == 255.) & 
                         (up_left == 255.) & (up_right == 255.) & 
                         (down_left == 255.) & (down_right == 255.))
        
        # Exclude boundaries from internal mask
        internal_mask[0, :] = False
        internal_mask[-1, :] = False
        internal_mask[:, 0] = False
        internal_mask[:, -1] = False
        
        # Update occupancy map
        occupancy_maps_gt[env_idx][four_neighbors_mask] = 255
        occupancy_maps_gt[env_idx][internal_mask] = 0
    
    # Save occupancy map
    if int(height) == height:
        occ_save_name = f"{dataset_name}_{grid_reso}_occ_map_height_{int(height)}_gt.pt"
    else:
        occ_save_name = f"{dataset_name}_{grid_reso}_occ_map_height_{str(height).replace(".", "d")}_gt.pt"  # like 1d5
    torch.save(occupancy_maps_gt, os.path.join(root_path, occ_save_name))
    print(f"Occupancy map saved to: {os.path.join(root_path, occ_save_name)}")
    
    # Visualization
    vis_path = os.path.join(root_path, "maps_visualization")
    os.makedirs(vis_path, exist_ok=True)
    
    for i in range(init_maps.shape[0]):
        # Visualize initial map
        init_map = init_maps[i].numpy()
        init_map = np.expand_dims(init_map, axis=-1)
        init_map = np.tile(init_map, (1, 1, 3))
        Image.fromarray(init_map.astype(np.uint8), mode="RGB").save(
            os.path.join(vis_path, f"init_map_{i}.png"))
        
        # Visualize occupancy map
        occ_map = occupancy_maps_gt[i].numpy()
        occ_map = np.expand_dims(occ_map, axis=-1)
        occ_map = np.tile(occ_map, (1, 1, 3))
        Image.fromarray(occ_map.astype(np.uint8), mode="RGB").save(
            os.path.join(vis_path, f"occupancy_map_{i}.png"))
    
    print(f"Visualizations saved to: {vis_path}")

def convert_to_h5(dataset_name: str, grid_reso: int = 128):
    """Step 5: Convert to HDF5 format"""
    print(f"\n=== Step 5: Convert to HDF5 format [{dataset_name}], resolution {grid_reso} ===")
    grid_path = os.path.join(OPEN_ROBOT_ROOT_DIR, 
        f"data_gleam/customized_data/gt/gt_{dataset_name}/{dataset_name}_{grid_reso}_4_grid_gt.pt")
    
    grid_4 = torch.load(grid_path)
    result_dict = tensor_to_dict(grid_4)
    
    save_path = os.path.join(OPEN_ROBOT_ROOT_DIR, 
        f"data_gleam/customized_data/gt/gt_{dataset_name}/{dataset_name}_{grid_reso}_grid_dict_gt.h5")
    
    # Save to HDF5 with compression
    with h5py.File(save_path, 'w') as f:
        for scene_idx, idx_coord in result_dict.items():
            idx_coord_np = idx_coord.cpu().numpy()
            dset = f.create_dataset(
                f'scene_{scene_idx}', 
                data=idx_coord_np,
                chunks=(1000, 6),
                compression="gzip",
                compression_opts=9
            )
            dset.attrs['num_voxels'] = len(idx_coord_np)
    
    print(f"Conversion complete! Saved to: {save_path}")

# ===== Main Control Function ===============================================

def main():
    DATASET_NAME = "YOUR_DATASET_NAME"  # Replace with your dataset name
    print(f"\n=== Starting Unified Preprocessing Pipeline for {DATASET_NAME} ===")

    # Define constants
    GRID_RESOLUTION = 128
    MAP_HEIGHT = 1.5  # Height for 2D map generation
    
    # Execute pipeline steps
    preprocess_meshes(DATASET_NAME)
    voxelize_scenes(DATASET_NAME, GRID_RESOLUTION)
    merge_voxel_data(DATASET_NAME, GRID_RESOLUTION)
    generate_2d_maps(DATASET_NAME, GRID_RESOLUTION, MAP_HEIGHT)
    convert_to_h5(DATASET_NAME, GRID_RESOLUTION)
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()