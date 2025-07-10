"""
This environment is based on drone env, while we ignore the drone control problem by directly setting the state of drone
"""
import os
import cv2
import json
import math
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
from collections import deque
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from gleam.env.env_base import ReconstructionDroneEnv
from gleam.utils.utils import pose_coord_to_2d_idx, pose_coord_to_2d_idx_vis
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder
import bfs_cuda_2D


class Env_GLEAM_Base(ReconstructionDroneEnv):
    def add_camera_to_actors(self):
        """
        Copied from legged_visual_input with slight modification on the camera pose (Z-axis + 0.1m)
        """
        if not self.cfg.return_visual_observation:
            return

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.cfg.visual_input.camera_width
        camera_properties.height = self.cfg.visual_input.camera_height
        camera_properties.horizontal_fov = self.cfg.visual_input.horizontal_fov
        camera_properties.far_plane = self.cfg.visual_input.far_plane
        camera_properties.near_plane = self.cfg.visual_input.near_plane
        camera_properties.supersampling_horizontal = self.cfg.visual_input.supersampling_horizontal
        camera_properties.supersampling_vertical = self.cfg.visual_input.supersampling_vertical
        camera_properties.enable_tensors = True

        camera_properties_col = gymapi.CameraProperties()
        camera_properties_col_reso = 40
        camera_properties_col.width = camera_properties_col_reso
        camera_properties_col.height = camera_properties_col_reso
        camera_properties_col.horizontal_fov = math.atan(camera_properties_col_reso/self.cfg.visual_input.camera_width) / math.pi * 180
        camera_properties_col.far_plane = self.cfg.visual_input.far_plane
        camera_properties_col.near_plane = self.cfg.visual_input.near_plane
        camera_properties_col.supersampling_horizontal = self.cfg.visual_input.supersampling_horizontal
        camera_properties_col.supersampling_vertical = self.cfg.visual_input.supersampling_vertical
        camera_properties_col.enable_tensors = True

        self.camera_handles = []
        self.camera_col_handles = []
        self.camera_angles = [0, 90, 180, 270]   # look around
        # self.camera_angles = [0]   # single cam, debug
        self.num_cam = len(self.camera_angles)

        for i in range(len(self.envs)):
            actor_handle = self.actor_handles[i]
            body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], actor_handle, 0)

            # Create 4 main cameras with different rotations
            for angle in self.camera_angles:
                cam_handle = self.gym.create_camera_sensor(self.envs[i], 
                                                           camera_properties)

                camera_offset = gymapi.Vec3(0, 0, 0.1)  # robot-mounted camera is 0.1m higher than robot
                camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.deg2rad(angle))

                self.gym.attach_camera_to_body(
                    cam_handle, 
                    self.envs[i], 
                    body_handle, 
                    gymapi.Transform(camera_offset, camera_rotation),
                    gymapi.FOLLOW_TRANSFORM
                )
                self.camera_handles.append(cam_handle)

            # collision camera
            cam_handle_col = self.gym.create_camera_sensor(self.envs[i], 
                                                           camera_properties_col)
            self.camera_col_handles.append(cam_handle_col)

        assert not set(self.camera_handles) & set(self.camera_col_handles), \
            "Two kinds of camera handles have overlapping elements"

    def _init_buffers(self):
        # load ground truth
        self._init_load_all()

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.skip = int(actor_root_state.shape[0] / self.num_envs)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[::self.skip, 0:3]
        self.base_quat = self.root_states[::self.skip, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_state = gymtorch.wrap_tensor(rigid_state).view(self.num_envs, -1, 13)

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.buffer_size = self.cfg.visual_input.stack
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # multi-scene setup
        self.env_to_scene = [env_idx * self.num_scene // self.num_envs \
                              for env_idx in range(self.num_envs)]
        self.env_to_scene = torch.tensor(self.env_to_scene, device=self.device)     # [num_env]

        # pose setup
        rpy_min = torch.tensor([0., 0., 0.], dtype=torch.float32, \
                               device=self.device).repeat(self.num_scene, 1)
        rpy_max = torch.tensor([0., 0., 2 * torch.pi], dtype=torch.float32, \
                               device=self.device).repeat(self.num_scene, 1)

        # clip pose for creating observation space
        # [num_scene, 6], (x_min, y_min, z_min, roll_min, pitch_min, yaw_min)
        if self.num_scene >= self.range_gt.shape[0]:
            self.clip_pose_world_low = torch.cat([self.range_gt.clone()[:, 1::2], \
                                                  rpy_min], dim=1).to(self.device)
            self.clip_pose_world_up = torch.cat([self.range_gt.clone()[:, ::2], \
                                                 rpy_max], dim=1).to(self.device)
        else:
            self.clip_pose_world_low = torch.cat([self.range_gt.clone()[:self.num_scene, 1::2], \
                                                  rpy_min], dim=1).to(self.device)
            self.clip_pose_world_up = torch.cat([self.range_gt.clone()[:self.num_scene, ::2], \
                                                 rpy_max], dim=1).to(self.device)

        if self.visualize_flag:
            self.pose_buf_vis = deque(maxlen=self.buffer_size)  # for visualization

        # action space
        self.actions = torch.tensor(self.cfg.normalization.init_action,
                                    dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.action_size = self.actions.shape[1]
        self.clip_actions_low = torch.tensor(self.cfg.normalization.clip_actions_low,
                                             dtype=torch.float32, device=self.device)
        self.clip_actions_up = torch.tensor(self.cfg.normalization.clip_actions_up,
                                            dtype=torch.float32, device=self.device)

        # reward functions
        assert self.buffer_size >= 2, "buffer size should be larger than 2"
        self.recent_num = 10     # early termination condition
        self.ratio_threshold_term = 0.98
        self.ratio_threshold_rew = 0.75
        self.collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reward_layout_ratio_buf = deque(maxlen=self.buffer_size)   # surface coverage ratio
        self.reward_layout_ratio_buf.extend(self.buffer_size * [torch.zeros(self.num_envs, device=self.device)])

        # only compute once
        self.blender2opencv = torch.FloatTensor([[1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]]).to(self.device)
        intrinsics = self.get_camera_intrinsics()   # [3, 3]
        self.inv_intri = torch.linalg.inv(intrinsics).to(self.device).to(torch.float32)

        height, width = self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width
        downsample_factor = 1
        xs = torch.linspace(0, width-downsample_factor, int(width/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys = torch.linspace(0, height-downsample_factor, int(height/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        norm_coord_pixel = torch.stack([xs, ys], dim=-1)    # [H, W, 2]
        norm_coord_pixel = torch.concat((norm_coord_pixel, 
                                         torch.ones_like(norm_coord_pixel[..., :1], device=self.device)), dim=-1).view(-1, 3)  # [H*W, 3], (u, v, 1)
        self.norm_coord_pixel_around = norm_coord_pixel.repeat(self.num_cam, 1)   # [num_cam*H*W, 3]

        # for visualization
        self.scanned_pc_coord = []
        self.scanned_pc_color = []
        self.transform_normdepth2PIL = ToPILImage()
        self.transforms_train_dict = dict()
        camera_properties = self.get_camera_properties()
        self.transforms_train_dict['h'] = self.cfg.visual_input.camera_height
        self.transforms_train_dict['w'] = self.cfg.visual_input.camera_width
        self.transforms_train_dict['camera_angle_x'] = camera_properties['horizontal_fov'] * np.pi / 180     # FOV, 90.0 degree -> 1.57
        self.transforms_train_dict['frames'] = []
        self._init_buffers_visual()

        # [num_scene] -> [num_env]
        self.range_gt_scenes = self.range_gt[self.env_to_scene]
        self.voxel_size_gt_scenes = self.voxel_size_gt[self.env_to_scene]
        self.num_valid_pixel_gt_scenes = self.num_valid_pixel_gt[self.env_to_scene]
        self.layout_maps_height_scenes = self.layout_maps_height[self.env_to_scene]
        self.move_range_scenes = self.move_range[self.env_to_scene]

        # del self.range_gt
        del self.voxel_size_gt
        del self.num_valid_pixel_gt
        del self.layout_maps_height
        del self.move_range

        # representations
        self.H_map = 128
        self.W_map = 128
        self.scanned_gt_map = torch.zeros(self.num_envs, self.H_map, self.W_map, 
                                          dtype=torch.float32, device=self.device)
        self.prob_map = torch.zeros(self.num_envs, self.H_map, self.W_map, 
                                    dtype=torch.float32, device=self.device)
        self.ego_prob_maps = torch.zeros(self.num_envs, self.H_map, self.W_map, 
                                         dtype=torch.float32, device=self.device)

    def _init_buffers_visual(self):
        if not self.cfg.return_visual_observation:
            return

        # only for visualization
        if self.visualize_flag:
            self.rgb_cam_tensors = []
            for i in range(self.num_envs):
                for cam_idx in range(self.num_cam):
                    im_rgb = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i*self.num_cam+cam_idx],
                                                            gymapi.IMAGE_COLOR)
                    torch_cam_tensor_rgb = gymtorch.wrap_tensor(im_rgb)
                    self.rgb_cam_tensors.append(torch_cam_tensor_rgb)

            self.rgb_cam_col_tensors = []
            for i in range(self.num_envs):
                im_rgb_col = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_col_handles[i],
                                                        gymapi.IMAGE_COLOR)
                torch_cam_tensor_rgb_col = gymtorch.wrap_tensor(im_rgb_col)
                self.rgb_cam_col_tensors.append(torch_cam_tensor_rgb_col)

        self.depth_cam_tensors = []
        for i in range(self.num_envs):
            for cam_idx in range(self.num_cam):
                im_depth = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i*self.num_cam+cam_idx],
                                                        gymapi.IMAGE_DEPTH)
                torch_cam_tensor_depth = gymtorch.wrap_tensor(im_depth)
                self.depth_cam_tensors.append(torch_cam_tensor_depth)

        self.depth_cam_col_tensors = []
        for i in range(self.num_envs):
            im_depth_col = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_col_handles[i],
                                                      gymapi.IMAGE_DEPTH)
            torch_cam_tensor_depth_col = gymtorch.wrap_tensor(im_depth_col)
            self.depth_cam_col_tensors.append(torch_cam_tensor_depth_col)

    def update_pose(self, ego_actions, e2w):
        """
        Update the pose of the robot in the world coordinate, including updating self.move_dist.
        Args:
            ego_actions: [num_env, 6], (x, y, z, roll, pitch, yaw), in egocentric coordinate. 
                         To unify the coordinate system, the policy predicts the normalized 
                         egocentric action in the range of [-1, 1].
            e2w: [num_env, 3, 3], egocentric to world coordinate
        """
        cur_poses = self.poses.clone()

        ego_move_xyz = ego_actions[:, :3].clone().to(torch.float32)    # [num_env, 3], z-axis = 0 for x-y action space
        ego_move_xyz[:, :2] = (ego_move_xyz[:, :2] - self.grid_size / 2) * self.voxel_size_gt_scenes[:, :2]   # world coordinate
        ego_move_xyz.unsqueeze_(-1)  # [num_env, 3, 1]

        world_move_xyz = torch.bmm(e2w, ego_move_xyz).squeeze(-1)   # [num_env, 3], world coordinate

        # Compute target poses by adding the world move to the current poses.
        tar_poses = self.poses.clone()
        tar_poses[:, :3] += world_move_xyz
        tar_pose_idx = pose_coord_to_2d_idx(
            poses=tar_poses[:, :2].clone(),    # last_pose_xy
            range_gt_scenes=self.range_gt_scenes,
            voxel_size_scenes=self.voxel_size_gt_scenes,
            map_size=self.grid_size).to(torch.long)

        # Update global poses for environments with no collision.
        self.tar_no_collision = self.scanned_gt_map[self.env_idx_tensor, tar_pose_idx[:, 0], tar_pose_idx[:, 1]] != 1.0
        self.poses[self.tar_no_collision, :3] += world_move_xyz[self.tar_no_collision]

        # Ensure the updated poses lie within map boundaries.
        self.clip_pose_map_bound()

        # Update movement distances after clipping.
        self.move_dist = self.poses[:, :3] - cur_poses[:, :3]


    def clip_pose_map_bound(self):
        self.poses = torch.clip(self.poses, self.clip_pose_world_low[self.env_to_scene], self.clip_pose_world_up[self.env_to_scene])
        self.poses[:, -1] = (self.poses[:, -1] + 2 * torch.pi) % (2 * torch.pi)  # wrap yaw angle to [0, 2*pi]

    def bfs_pathfinding(self, occupancy_maps, starts, goals):
        """
        Perform BFS-based pathfinding using a custom CUDA extension.

        Parameters:
            occupancy_maps: torch.Tensor of shape [num_env, H, W] (dtype=torch.int32, on CUDA)
            starts: torch.Tensor of shape [num_env, 2] (dtype=torch.int32, on CUDA)
            goals: torch.Tensor of shape [num_env, 2] (dtype=torch.int32, on CUDA)

        Returns:
            path_lengths: torch.Tensor of shape [num_env] containing path lengths
        """
        path_lengths = torch.full((occupancy_maps.size(0),), -1.0, dtype=torch.float32, device=self.device)
        bfs_cuda_2D.BFS_CUDA_2D(occupancy_maps, starts, goals, path_lengths)
        return path_lengths

    def save_all_img(self):
        """
        self.depth_processed: [num_env, num_cam, H, W]
        self.rgb_processed: [num_env, num_cam, H, W, 3]
        """
        # video demo
        save_path = self.save_path
        save_path_depth = save_path + '/depth'
        save_path_depth_col = save_path + '/depth_col'
        save_path_depth_vis = save_path + '/depth_vis_wo_seg'
        save_path_rgb = save_path + '/rgb'
        save_path_rgb_col = save_path + '/rgb_col'
        os.makedirs(save_path_depth, exist_ok=True)
        os.makedirs(save_path_depth_col, exist_ok=True)
        os.makedirs(save_path_depth_vis, exist_ok=True)
        os.makedirs(save_path_rgb, exist_ok=True)
        os.makedirs(save_path_rgb_col, exist_ok=True)

        for i in range(self.num_envs):
            if i != self.vis_obj_idx:
                continue

            idx = int(self.cur_episode_length[self.vis_obj_idx].item())

            num_around = self.depth_processed.shape[1]
            for heading_idx in range(num_around):   # 0: original heading, 1: 90, 2: 180, 3: 270
                # depth
                fname = os.path.join(save_path_depth, f"depth_{idx}_{heading_idx}.tiff")
                depth_cam_img = self.depth_processed[i, heading_idx].clone()
                depth_cam_img = depth_cam_img.cpu().detach().numpy()
                depth_cam_img_PIL = self.transform_normdepth2PIL(depth_cam_img)  # range: [0., 8.]
                depth_cam_img_PIL.save(fname)

                # depth_vis_wo_seg
                fname = os.path.join(save_path_depth_vis, f"depth_{idx}_{heading_idx}.png")
                depth_cam_img_vis = self.depth_processed[i, heading_idx].clone()
                depth_cam_img_vis = depth_cam_img_vis.cpu().detach().numpy()
                depth_cam_img_vis = (depth_cam_img_vis / depth_cam_img_vis.max()) * 255
                depth_cam_img_vis = np.expand_dims(depth_cam_img_vis, axis=-1)
                depth_cam_img_vis = np.tile(depth_cam_img_vis, (1,1,3))
                depth_cam_img_vis = Image.fromarray(depth_cam_img_vis.astype(np.uint8), mode="RGB")
                depth_cam_img_vis.save(fname)

                # rgb
                fname = os.path.join(save_path_rgb, f"rgb_{idx}_{heading_idx}.png")
                cam_img = self.rgb_processed[i, heading_idx].clone() # [H, W, 3], range:[0, 1]
                cam_img = cam_img.cpu().detach().numpy()
                image = Image.fromarray(cam_img.astype(np.uint8), mode="RGB")
                image.save(fname)

            # depth_col
            fname = os.path.join(save_path_depth_col, f"depth_col_{idx}.tiff")
            depth_cam_img_col = self.depth_processed_col[self.vis_obj_idx].clone()
            depth_cam_img_col = depth_cam_img_col.cpu().detach().numpy()
            depth_cam_img_col_PIL = self.transform_normdepth2PIL(depth_cam_img_col)  # range: [0., 8.]
            depth_cam_img_col_PIL.save(fname)

            # rgb_col
            fname = os.path.join(save_path_rgb_col, f"rgb_col_{idx}.png")
            cam_img_col = self.rgb_processed_col[self.vis_obj_idx].clone() # [H, W, 3], range:[0, 1]
            cam_img_col = cam_img_col.cpu().detach().numpy()
            image = Image.fromarray(cam_img_col.astype(np.uint8), mode="RGB")
            image.save(fname)

        self.transform_matrix = dict()
        self.transform_matrix['transform_matrix'] = self.c2ws[self.vis_obj_idx, 0].tolist()     # For look around setup, only store the first camera pose
        self.transform_matrix['pose'] = self.poses.tolist()[self.vis_obj_idx]   # [num_env, pose_size]
        self.transform_matrix['coverage_ratio'] = round(self.reward_layout_ratio_buf[-1][[self.vis_obj_idx]].item(), 3)

        self.transforms_train_dict['frames'].append(self.transform_matrix)
        content = json.dumps(self.transforms_train_dict, indent=4, sort_keys=True)    # dict is dumped into str

        f_train = open(self.save_path+'/transforms_depth.json', 'w')
        f_train.write(content)
        f_train.close()

        print("cur_episode_lendth: ", int(self.cur_episode_length[self.vis_obj_idx].item()))

    def save_img_poses(self):
        # if self.reset_once_flag[self.vis_obj_idx] == 1:
        #     exit()

        save_path = self.save_path
        save_path_depth = save_path + '/depth'
        save_path_depth_vis = save_path + '/depth_vis_wo_seg'
        os.makedirs(save_path_depth, exist_ok=True)

        for i in range(self.num_envs):
            if i != self.vis_obj_idx:
                continue

            idx = int(self.cur_episode_length[self.vis_obj_idx].item())

            num_around = self.depth_processed.shape[1]
            for heading_idx in range(num_around):   # 0: original heading, 1: 90, 2: 180, 3: 270
                # depth
                fname = os.path.join(save_path_depth, f"depth_{idx}_{heading_idx}.tiff")
                depth_cam_img = self.depth_processed[i, heading_idx].clone()
                depth_cam_img = depth_cam_img.cpu().detach().numpy()
                depth_cam_img_PIL = self.transform_normdepth2PIL(depth_cam_img)  # range: [0., 8.]
                depth_cam_img_PIL.save(fname)

                # depth_vis_wo_seg
                fname = os.path.join(save_path_depth_vis, f"depth_{idx}_{heading_idx}.png")
                depth_cam_img_vis = self.depth_processed[i, heading_idx].clone()
                depth_cam_img_vis = depth_cam_img_vis.cpu().detach().numpy()
                depth_cam_img_vis = (depth_cam_img_vis / depth_cam_img_vis.max()) * 255
                depth_cam_img_vis = np.expand_dims(depth_cam_img_vis, axis=-1)
                depth_cam_img_vis = np.tile(depth_cam_img_vis, (1,1,3))
                depth_cam_img_vis = Image.fromarray(depth_cam_img_vis.astype(np.uint8), mode="RGB")
                depth_cam_img_vis.save(fname)

        self.transform_matrix = dict()
        self.transform_matrix['transform_matrix'] = self.c2ws[self.vis_obj_idx, 0].tolist()     # For look around setup, only store the first camera pose
        self.transform_matrix['pose'] = self.poses.tolist()[self.vis_obj_idx]   # [num_env, pose_size]
        self.transform_matrix['coverage_ratio'] = round(self.reward_layout_ratio_buf[-1][[self.vis_obj_idx]].item(), 3)

        self.transforms_train_dict['frames'].append(self.transform_matrix)
        content = json.dumps(self.transforms_train_dict, indent=4, sort_keys=True)    # dict is dumped into str

        f_train = open(self.save_path+'/trajectory.json', 'w')
        f_train.write(content)
        f_train.close()

    def _reset_root_states(self, env_ids):
        """ Resets the root states of agents in envs to be reseted
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[::self.skip][env_ids] = self.base_init_state
            self.root_states[::self.skip][env_ids, :3] += self.env_origins[env_ids]
            self.root_states[::self.skip][env_ids, :2] += torch_rand_float(
                -1., 1., (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:   # <-
            self.root_states[::self.skip][env_ids] = self.base_init_state
            self.root_states[::self.skip][env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.clone().to(dtype=torch.int32) * self.skip
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def back_projection_stack(self, downsample_factor=1, visualize=False):
        """
        self.depth_processed: [num_env, num_cam, H, W]
        """
        num_env = self.depth_processed.shape[0]
        num_cam = self.depth_processed.shape[1]
        H, W = self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width

        depth_maps = self.depth_processed[:, :, ::downsample_factor, ::downsample_factor]   # [num_env, num_cam, H_down, W_down]

        # back-projection
        depth_maps = depth_maps.reshape(num_env, -1)          # [num_env, num_point], where num_point == num_cam * H * W
        coords_pixel = torch.einsum('ij,jk->ijk', depth_maps, self.norm_coord_pixel_around)   # [num_env, num_point, 3]

        # inv_intri: [3, 3], coords_pixel: [num_env, num_point, 3]
        coords_cam = torch.einsum('ij,nkj->nki', self.inv_intri, coords_pixel)    # [num_env, num_point, 3]
        coords_cam_homo = torch.concat((coords_cam, torch.ones_like(coords_cam[..., :1], device=self.device)), dim=-1)   # [num_env, num_point, 4], where num_point == num_cam * H * W

        # Stack all transformation matrices
        if isinstance(self.c2ws, torch.Tensor):
            c2ws = self.c2ws
        elif isinstance(self.c2ws, list):
            c2ws = torch.stack(self.c2ws, dim=1)  # [num_env, num_cam, 4, 4]
        
        coords_cam_homo = coords_cam_homo.view(num_env, num_cam, H*W, 4)
        coords_world_around = torch.matmul(
            c2ws.unsqueeze(2),  # [num_env, num_cam, 1, 4, 4]
            coords_cam_homo.unsqueeze(-1)  # [num_env, num_cam, H*W, 4, 1]
        ).squeeze(-1)  # [num_env, num_cam, H*W, 4]

        coords_world_around = coords_world_around[..., :3]  # [num_env, num_cam, H*W, 3]
        coords_world_around = coords_world_around.reshape(num_env, num_cam*H*W, 3)

        if visualize:
            rgb_maps = self.rgb_processed.clone()[:, :, ::downsample_factor, ::downsample_factor]   # [num_env, H_down, W_down]
            colors_world_around = rgb_maps[..., :3].reshape(self.num_envs, -1, 3)	# [num_env, num_cam*H*W, 3]
            return coords_world_around, colors_world_around    # [num_env, num_point, 3]
        return coords_world_around

    def local_path_astar(self, occ_map, start_idx, end_idx):
        """
        Args:
            matrix: [H, W] matrix representing the occupancy grid, where 1 is free and 0 is occupied.
        """
        # print(f'start = {start_idx}, end = {end_idx}')

        occ_map_binary = (occ_map < 0).float()
        occ_grid = Grid(matrix=occ_map_binary.cpu().numpy())

        start = occ_grid.node(start_idx[1], start_idx[0])  # pathfinding expects (x, y)
        end = occ_grid.node(end_idx[1], end_idx[0])

        # use A* algorithm to find the path
        # finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        # finder = AStarFinder(max_runs=50, diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, occ_grid)

        # print('path length:', len(path))
        return path, occ_map_binary

    def check_termination(self):
        """ Check if environments need to be reset
        Termination conditions:
            1. collision
            2. meaningless wander
            3. steps == max_episode_length
            4. coverage ratio threshold
        """
        # collision, including rigid body collision and depth collision
        self.reset_buf = self.collision_flag.clone()

        # meaningless wander
        recent_CR_thres = 0.01

        recent_CR = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-self.recent_num]  # [num_envs]
        meaningless_wander = (recent_CR < recent_CR_thres)  # [num_envs]
        meaningless_wander *= (self.cur_episode_length > self.recent_num)
        self.reset_buf |= meaningless_wander.clone()

        # max_step
        # self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.termination.max_step_done:  # <-
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)    # [num_envs]
            self.reset_buf |= self.time_out_buf

        # coverage ratio > threshold
        last_ratio = self.reward_layout_ratio_buf[-1]
        self.reset_buf |= (last_ratio > self.ratio_threshold_term)

    def get_camera_view_matrix(self):
        """
        return Extrinsics.t() instead of Extrinsics. E * P = P * E.t()
        """
        ret = []
        for k, handle in enumerate(self.camera_handles):
            ret.append(self.gym.get_camera_view_matrix(self.sim, self.envs[k//self.num_cam], handle))
        return torch.FloatTensor(np.stack(ret)).to(self.device)

    def draw_rainbow_path(self, local_nodes, gt_map):
        """
        Draw path with color gradient: red → orange → yellow → green → blue → purple
        Args:
            local_nodes: tensor of shape [N, 2] containing (x, y) coordinates
            gt_map: tensor of shape [H, W, 3] for RGB values
        """
        num_points = local_nodes.shape[0]
        
        # Create color gradient
        colors = torch.zeros((num_points, 3), device=local_nodes.device)
        t = torch.linspace(0, 1, num_points, device=local_nodes.device)
        
        # Red to purple through rainbow
        colors[:, 0] = 255 * (t < 0.5)  # Red: 255 -> 0
        colors[:, 1] = 255 * torch.sin(np.pi * t)  # Green: 0 -> 255 -> 0
        colors[:, 2] = 255 * (t > 0.5)  # Blue: 0 -> 255
        
        # Draw path
        gt_map[local_nodes[:, 1], local_nodes[:, 0]] = colors

        return gt_map

    def visualize_all_2d_occ_map_eval_local(self):
        """ Visualize all 2D occupancy maps for evaluation. 
            Don't call visualize_2D_occ_map() at the same time.
        """
        save_path_occ = os.path.join(self.save_path, "all_2D_occ_map_local")
        os.makedirs(save_path_occ, exist_ok=True)

        self.pose_buf_vis.extend([self.poses.clone()])
        current_poses = torch.stack(tuple(self.pose_buf_vis), dim=1)    # [num_env, num_step, pose_size]

        for env_idx in range(self.num_envs):
            if self.reset_once_flag[env_idx] or self.cur_episode_length[env_idx] == 0 :
                continue
            # update local path
            start_idx = self.poses_idx_old[env_idx].cpu().tolist()  # [H_idx, W_idx]
            end_idx = self.poses_idx[env_idx].cpu().tolist()

            local_path, _ = self.local_path_astar(occ_map=self.occ_maps_tri_cls[env_idx],
                                                    start_idx=start_idx,
                                                    end_idx=end_idx)

            nodes_from_local_path = torch.tensor(
                [[node.x, node.y] for node in local_path],
                dtype=torch.long,
                device=self.device
            )
            self.local_paths[env_idx].append(nodes_from_local_path)
            # print("A* CPU: ", local_path)

            if self.reset_buf[env_idx] and self.reset_once_flag[env_idx] == False:
                # 2D scanned gt map
                all_xy_world = current_poses[env_idx][:, :2]    # [num_step, 2]

                # [num_step, 2]
                pose_idx = pose_coord_to_2d_idx_vis(poses=all_xy_world,
                                                        range_gt_scenes=self.range_gt_scenes[env_idx],
                                                        voxel_size_scenes=self.voxel_size_gt_scenes[env_idx])

                gt_map = self.layout_maps_height_scenes[env_idx].clone().cpu()
                gt_idx = (gt_map == 1.)  # [256, 256]
                gt_map = gt_map.unsqueeze(-1).repeat(1, 1, 3) # [256, 256, 3]
                gt_map[gt_idx] = torch.tensor([255, 0, 0], dtype=torch.float32)

                scanned_map = self.scanned_gt_map[env_idx].clone().cpu()   # [H, W]
                gt_map[scanned_map == 1.] = torch.tensor([255, 255, 255], dtype=torch.float32)
                gt_map[pose_idx[:, 0], pose_idx[:, 1]] = torch.tensor([30, 144, 255], dtype=torch.float32)

                local_nodes = torch.cat(self.local_paths[env_idx], dim=0).cpu()
                if len(local_nodes) > 0:
                    gt_map = self.draw_rainbow_path(local_nodes, gt_map)

                gt_map = gt_map.cpu().numpy().astype(np.uint8)
                pose_idx = pose_idx.cpu().numpy()

                # Mark the points themselves
                for point in pose_idx:
                    cv_point = (int(point[1]), int(point[0]))
                    cv2.circle(gt_map, cv_point, radius=2, color=(30, 144, 255), thickness=-1)

                scanned_gt_map = Image.fromarray(gt_map, mode="RGB")
                scanned_gt_map.save(f"{save_path_occ}/scene_{env_idx}.png")
                print(f"Save scene_{env_idx}_step_{int(self.cur_episode_length[env_idx].item())}_gt.png")

                self.reset_once_flag[env_idx] = True
                self.reset_once_cr[env_idx] = self.reward_layout_ratio_buf[-1][env_idx].item()

        if self.reset_once_flag.sum() == self.num_envs:
            print(self.reset_once_cr.mean())
            occ_map_pngs = os.listdir(save_path_occ)
            occ_map_pngs = [occ_map_png for occ_map_png in occ_map_pngs \
                            if occ_map_png.startswith('scene_') and occ_map_png.endswith('.png')]
            occ_map_pngs.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))

            fig, axes = plt.subplots(4, 8, figsize=(20, 10))
            axes = axes.flatten()
            for env_idx in range(self.num_envs):
                img_name = f"scene_{env_idx}.png"
                img_file = os.path.join(save_path_occ, img_name)
                
                if os.path.exists(img_file):
                    # Read and plot image
                    img = plt.imread(img_file)
                    axes[env_idx].imshow(img)
                    axes[env_idx].set_title(f'Scene_{env_idx}, CR: {round(self.reset_once_cr[env_idx].item() * 100, 2)}%')
                    axes[env_idx].axis('off')
                else:
                    print(f"Warning: {img_name} not found")
                    
            # Adjust layout and save figure
            plt.tight_layout()
            save_path_occ_png = os.path.join(save_path_occ, 'all_2D_occ_map_eval.png')
            plt.savefig(save_path_occ_png, bbox_inches='tight', dpi=300)
            plt.close()
            
            print("Done")
            exit()
