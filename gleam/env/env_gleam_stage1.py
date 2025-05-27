from gym.spaces import Box, Dict, MultiDiscrete
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym import OPEN_ROBOT_ROOT_DIR
import os
import torch
import random
import numpy as np
from collections import deque
from gleam.env.env_gleam_base import Env_GLEAM_Base
from gleam.utils.utils import scanned_pts_to_2d_idx, pose_coord_to_2d_idx, \
                        bresenham_2d, discretize_prob_map, \
                        extract_ego_maps, compute_frontier_map, \
                        create_e2w_from_poses


class Env_GLEAM_Stage1(Env_GLEAM_Base):
    def __init__(self, *args, **kwargs):
        """
        Training set of stage 1 includes 512 indoor scenes from:
            procthor_4-room_256,
            procthor_5-room_164,
            procthor_8-room-3-bed_28, 
            procthor_12-room-3-bed_32,
            hssd_32.
        """
        self.visualize_flag = False     # training
        # self.visualize_flag = True    # visualization

        # self.num_scene = 4    # debug
        # print("*"*50, "num_scene: ", self.num_scene, "*"*50)

        self.num_scene = 512

        super(Env_GLEAM_Base, self).__init__(*args, **kwargs)

    def _additional_create(self, env_handle, env_index):
        """
        If there are N training scenes and M environments, each environment will load N//M scenes.
        Only the first scene (idx == 0) is active, and the others are inactive.
        """
        assert self.cfg.return_visual_observation, "visual observation should be returned!"
        assert self.num_scene >= self.num_envs, "num_scene should be larger than num_envs"

        # urdf load, create actor
        dataset_name = "stage1_512"
        urdf_path = f"data_gleam/train_{dataset_name}/urdf"

        scene_per_env = self.num_scene // self.num_envs

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        inactive_x = self.env_origins[:, 0].max() + 15.
        inactive_y = self.env_origins[:, 1].max() + 15.
        inactive_z = self.env_origins[:, 2].max()

        for idx in range(scene_per_env):
            scene_idx = env_index * scene_per_env + idx
            urdf_name = f"scene_{scene_idx}.urdf"
            asset = self.gym.load_asset(self.sim, urdf_path, urdf_name, asset_options)

            pose = gymapi.Transform()
            if idx == 0:    # Only the first scene (idx == 0) is active
                pose.p = gymapi.Vec3(self.env_origins[env_index][0],
                                        self.env_origins[env_index][1],
                                        self.env_origins[env_index][2])
            else:
                pose.p = gymapi.Vec3(inactive_x, inactive_y, inactive_z)
            pose.r = gymapi.Quat(-np.pi/2, 0.0, 0.0, np.pi/2)
            self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)

        self.additional_actors[env_index] = [i+1 for i in range(scene_per_env)]

    def _init_load_all(self):
        """
        Load all ground truth data.
        """
        self.grid_size = 128
        self.motion_height = 1.5    # 1.5m
        dataset_name = "stage1_512"
        gt_path = os.path.join(OPEN_ROBOT_ROOT_DIR, f"data_gleam/train_{dataset_name}/gt/")

        # [num_scene, 3]
        self.voxel_size_gt = torch.load(gt_path+f"{dataset_name}_{self.grid_size}_voxel_size_gt.pt", 
                                        map_location=self.device)[:self.num_scene]

        # [num_scene, 6], (x_max, x_min, y_max, y_min, z_max, z_min)
        self.range_gt = torch.load(gt_path+f"{dataset_name}_{self.grid_size}_range_gt.pt", 
                                    map_location=self.device)[:self.num_scene]

        # [num_scene, grid_size, grid_size], layout map at the height of {self.motion height}
        self.layout_maps_height = torch.load(gt_path+f"{dataset_name}_{self.grid_size}_occ_map_height_1d5_gt.pt", 
                                            map_location=self.device)[:self.num_scene].to(torch.float16)
        self.layout_maps_height /= 255.

        # [num_scene]
        self.num_valid_pixel_gt = self.layout_maps_height.sum(dim=(1, 2))

        # [num_scene, 128, 128]
        init_maps = torch.load(gt_path+f"{dataset_name}_{self.grid_size}_init_map_1d5.pt", 
                                map_location=self.device)[:self.num_scene]
        init_maps /= 255.

        # len() == self.num_scene, the shape of element is [num_non_zero_pixel, 2]
        self.init_maps_list = [(torch.nonzero(init_maps[idx]) / (self.grid_size - 1) * 2 - 1)\
                                * self.range_gt[idx, :4:2]
                               for idx in range(self.num_scene)]

        print("Loaded all ground truth data.")

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
            self.clip_pose_world_low = torch.cat([self.range_gt[:, 1::2], \
                                                  rpy_min], dim=1).to(self.device)
            self.clip_pose_world_up = torch.cat([self.range_gt[:, ::2], \
                                                 rpy_max], dim=1).to(self.device)
        else:
            self.clip_pose_world_low = torch.cat([self.range_gt[:self.num_scene, 1::2], \
                                                  rpy_min], dim=1).to(self.device)
            self.clip_pose_world_up = torch.cat([self.range_gt[:self.num_scene, ::2], \
                                                 rpy_max], dim=1).to(self.device)

        if self.visualize_flag:
            self.pose_buf_vis = deque(maxlen=self.buffer_size)  # for visualization

        # action space
        self.actions = torch.tensor(self.cfg.normalization.init_action, 
                                    dtype=torch.int64, device=self.device).repeat(self.num_envs, 1)
        self.action_size = self.actions.shape[1]
        self.clip_actions_low = torch.tensor(self.cfg.normalization.clip_actions_low, 
                                             dtype=torch.int64, device=self.device)
        self.clip_actions_up = torch.tensor(self.cfg.normalization.clip_actions_up, 
                                            dtype=torch.int64, device=self.device)

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

        self._init_buffers_visual()

        # [num_scene] -> [num_env]
        self.range_gt_scenes = self.range_gt[self.env_to_scene]
        self.voxel_size_gt_scenes = self.voxel_size_gt[self.env_to_scene]
        self.num_valid_pixel_gt_scenes = self.num_valid_pixel_gt[self.env_to_scene]
        self.layout_maps_height_scenes = self.layout_maps_height[self.env_to_scene]

        # pose setup
        init_poses = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        for env_idx in range(self.num_envs):
            scene_idx = self.env_to_scene[env_idx]
            init_poses[env_idx, :2] = self.init_maps_list[scene_idx][random.randint(0, len(self.init_maps_list[scene_idx])-1)]

        init_poses[:, 2] = self.motion_height
        self.poses = init_poses.clone()
        self.pose_size = self.poses.shape[1]
        self.poses_idx = torch.tensor([self.grid_size//2, self.grid_size//2], 
                                      dtype=torch.int32, device=self.device).repeat(self.num_envs, 1)

        self.pose_buf = []
        self.pose_buf += 10 * [self.poses.clone()]
        # [num_env, buffer_size, action_size]
        self.world_pose_buf = torch.zeros(self.num_envs, self.buffer_size, self.pose_size, 
                                            dtype=torch.float32, device=self.device)
        # [num_env, buffer_size, action_size]
        self.ego_pose_buf = torch.zeros(self.num_envs, self.buffer_size, self.pose_size,
                                        dtype=torch.float32, device=self.device)

        self.map_size_tensor = torch.tensor([self.grid_size-1, self.grid_size-1], device=self.device)
        self.env_idx_tensor = torch.arange(self.num_envs, device=self.device)

        # map representations
        self.scanned_gt_map = torch.zeros(self.num_envs, self.grid_size, self.grid_size, 
                                          dtype=torch.float32, device=self.device)
        self.prob_map = torch.zeros(self.num_envs, self.grid_size, self.grid_size, 
                                    dtype=torch.float32, device=self.device)
        self.ego_prob_maps = torch.zeros(self.num_envs, self.grid_size, self.grid_size, 
                                         dtype=torch.float32, device=self.device)
        self.occ_maps_tri_cls = torch.zeros(self.num_envs, self.grid_size, self.grid_size, 
                                            dtype=torch.float32, device=self.device)

        # define a 3x3 convolutional kernel to check 4-connected neighbors
        self.frontier_kernel = torch.tensor([[0, 1, 0],
                                            [1, 0, 1],
                                            [0, 1, 0]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # scene updating strategy
        self.scene_per_env = self.num_scene // self.num_envs
        self.inactive_xyz = (self.env_origins[:, 0].max() + 15.,
                             self.env_origins[:, 1].max() + 15.,
                             self.env_origins[:, 2].max())
        self.inactive_xyz = torch.tensor(self.inactive_xyz, device=self.device)
        self.active_scene_ids = self.env_to_scene.tolist()
        self.inactive_scene_ids = [scene_idx for scene_idx in range(self.num_scene) \
                                   if scene_idx not in self.active_scene_ids]

        self.ego_cell_size = 0.1

        # visualization
        if self.visualize_flag:
            # visualize scenes
            self.vis_obj_idx = 0
            print("Visualization object index: ", self.vis_obj_idx)

            self.reset_once_flag = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.reset_once_cr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.local_paths = [[] for _ in range(self.num_envs)]

            self.save_path = f'./gleam/scripts/video/train_stage1_512'
            os.makedirs(self.save_path, exist_ok=True)

    def reset(self):
        """ Initialization: Reset all envs"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # initial pose
        self.actions = torch.clip(self.actions, self.clip_actions_low, 
                                  self.clip_actions_up).to(self.device)
        self.e2w = create_e2w_from_poses(self.poses, self.device)  # [num_env, 4, 4]

        self.update_pose(ego_actions=self.actions, e2w=self.e2w)

        self.render()
        self.set_state(self.poses)
        obs = self.post_physics_step(if_reset=True)
        return obs

    def step(self, actions):
        """
        Set the position (x, y, z) and orientation (r, p, y) for the camera.
        """
        self.actions = torch.clip(actions, self.clip_actions_low, self.clip_actions_up)

        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if len(env_ids) != 0:
            self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action, 
                                                 dtype=torch.int64, device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]
            self.e2w[env_ids] = create_e2w_from_poses(self.poses[env_ids], self.device)    # [num_env, 3, 3]

        current_poses = self.poses.clone()
        self.update_pose(ego_actions=self.actions, e2w=self.e2w)    # update self.poses in world coordinate

        current_move_xyz = torch.cat([self.move_dist, 
                                      torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)], dim=1) # [num_env, 3]
        self.set_collision_cam_pose(current_poses, current_move_xyz)  # set the pose of collision camera

        self.render()
        self.set_state(self.poses)
        obs, rewards, dones, infos = self.post_physics_step()
        return obs, rewards, dones, infos

    def update_observation(self):
        self.update_pose_buf()
        self.update_occ_map_2d()

    def update_pose_buf(self):
        """
        Args:
            self.poses: [num_env, pose_size]
            self.world_pose_buf: [num_env, buffer_size, pose_size]
            self.ego_pose_buf: [num_env, buffer_size, pose_size]
        """
        actual_poses = self.poses.clone()

        for env_idx in range(self.num_envs):
            num_step = int(self.cur_episode_length[env_idx].item())
            if num_step != 0 and num_step <= self.buffer_size - 1:
                self.world_pose_buf[env_idx, 1:num_step+1] = self.world_pose_buf[env_idx, :num_step].clone()
            else:
                self.world_pose_buf[env_idx, 1:] = self.world_pose_buf[env_idx, :self.buffer_size-1].clone()

            self.world_pose_buf[env_idx, 0] = actual_poses[env_idx]
            self.ego_pose_buf[env_idx, :num_step+1] = self.world_pose_buf[env_idx, :num_step+1] - self.world_pose_buf[env_idx, 0].clone()

        self.pose_buf.pop(0)
        self.pose_buf.append(actual_poses.clone())

    def update_occ_map_2d(self):
        # Update camera view matrices
        extrinsics = self.get_camera_view_matrix() # [num_env*num_cam, 4, 4]
        c2w = torch.linalg.inv(extrinsics.transpose(-2, -1)) @ self.blender2opencv.unsqueeze(0) # [num_env*num_cam, 4, 4]
        c2w = c2w.reshape(self.num_envs, self.num_cam, 4, 4)
        c2w[:, :, :3, 3] -= self.env_origins.unsqueeze(1)
        self.c2ws = c2w # [num_env, num_cam, 4, 4]

        if self.visualize_flag:
            pts_target, color_target = self.back_projection_stack(downsample_factor=1, visualize=True)   # [num_env, num_point, 3], num_point == H * W
            # self.scanned_pc_coord.append(pts_target[self.vis_obj_idx].cpu().clone())
            # self.scanned_pc_color.append(color_target[self.vis_obj_idx].cpu().clone())
        else:
            pts_target = self.back_projection_stack(downsample_factor=1)   # [num_env, num_point, 3], where num_point == num_stack * H * W

        # list of (num_valid_pts_idx, 2), at the height of {self.motion height}
        pts_idx_all = scanned_pts_to_2d_idx(pts_target=pts_target,
                                                range_gt_scenes=self.range_gt_scenes,
                                                voxel_size_scenes=self.voxel_size_gt_scenes,
                                                motion_height=self.motion_height,
                                                map_size=self.grid_size)

        # [num_env, 2]
        self.poses_idx_old = self.poses_idx.to(torch.int32).clone()
        pose_idx = pose_coord_to_2d_idx(poses=self.poses[:, :2].clone(),    # last_pose_xy
                                            range_gt_scenes=self.range_gt_scenes,
                                            voxel_size_scenes=self.voxel_size_gt_scenes,
                                            map_size=self.grid_size)
        self.poses_idx = pose_idx.to(torch.int32).clone()


        current_pose_idx = self.poses_idx.to(torch.long)    # [num_env, 2]
        batch_idx = torch.arange(self.num_envs) # [num_env]
        self.current_pose_state = self.occ_maps_tri_cls[batch_idx, current_pose_idx[:, 0], current_pose_idx[:, 1]].clone()   # [num_env]


        self.e2w = create_e2w_from_poses(poses=self.poses,
                                        device=self.device)    # [num_env, 3, 3]

        for env_idx in range(self.num_envs):
            pts_idx = pts_idx_all[env_idx]

            if (isinstance(pts_idx, list) and len(pts_idx) == 0) or pts_idx.shape[0] == 0:
                continue

            # [num_point, 2]
            ray_cast_paths = bresenham_2d(pts_source=pose_idx[env_idx: env_idx+1],
                                                pts_target=pts_idx,
                                                map_size=self.grid_size)

            self.prob_map[env_idx, ray_cast_paths[:, 0], ray_cast_paths[:, 1]] -= 0.05
            self.prob_map[env_idx, pts_idx[:, 0], pts_idx[:, 1]] = 1.0


        # occ_maps: [num_env, H, W] in world coordinate, where {1: occupied, 0: free/unknown}
        # occ_maps_tri_cls: [num_env, H, W] in world coordinate, where {-1: free, 0: unknown, 1: occupied}
        occ_maps, self.occ_maps_tri_cls = discretize_prob_map(self.prob_map,
                                                                threshold_occu=0.5,
                                                                threshold_free=0.0)

        # [num_env, H, W], Update scanned_gt_map for computing coverage reward
        self.scanned_gt_map = torch.clip(
            self.scanned_gt_map + occ_maps * self.layout_maps_height_scenes,
            max=1, min=0
        )


        # [num_env, H, W], Update ego_prob_maps that serves as representation
        ego_prob_maps = extract_ego_maps(global_maps=self.occ_maps_tri_cls.clone(),
                                                cell_sizes=self.voxel_size_gt_scenes[:, :2],
                                                poses_idx=current_pose_idx,
                                                ego_cm=self.ego_cell_size)

        # self.visualize_global_and_ego_maps(global_maps=self.occ_maps_tri_cls.clone(),
        #                                     ego_maps=ego_prob_maps.clone(),
        #                                     poses_idx=current_pose_idx,
        #                                     cell_sizes=self.voxel_size_gt_scenes[:, :2],
        #                                     ego_cm=self.ego_cell_size)

        # [num_env, H, W], recognize frontier
        self.ego_prob_maps = ego_prob_maps.clone()

        ego_occ_masks = (ego_prob_maps != 1.0).to(torch.bool)
        ego_frontier_masks = compute_frontier_map(ego_prob_maps=ego_prob_maps, frontier_kernel=self.frontier_kernel)
        ego_frontier_masks = ego_frontier_masks & ego_occ_masks

        self.ego_prob_maps[ego_frontier_masks] = 2.0

    def check_motion_collision_local_2d(self):
        """ check collision in motion, including rigid body collision and local planning collision"""

        # # collision by contact force
        # self.collision_rigid = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.collision_rigid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cur_pose_idx = self.poses_idx.to(torch.long)
        for env_idx in range(self.num_envs):
            self.collision_rigid[env_idx] = self.layout_maps_height_scenes[env_idx, cur_pose_idx[env_idx, 0], cur_pose_idx[env_idx, 1]] == 1.0

        # collision by local planner
        height_c = int(self.depth_processed_col.shape[1] / 2) - 1
        width_c = int(self.depth_processed_col.shape[2] / 2) - 1
        # [num_env, 4]
        depth_center_area = torch.stack([self.depth_processed_col[:, height_c, width_c],
                                         self.depth_processed_col[:, height_c, width_c+1],
                                         self.depth_processed_col[:, height_c+1, width_c],
                                         self.depth_processed_col[:, height_c+1, width_c+1]], dim=1)
        depth_center_area = torch.abs(depth_center_area)
        min_vis_dist, _ = torch.min(depth_center_area, dim=1)  # [num_env]
        
        move_dist = torch.norm(self.move_dist, dim=-1)
        self.collision_vis = (min_vis_dist < move_dist - 0.15)
        self.collision_vis[self.cur_episode_length == 0] = False

        if self.visualize_flag:
            self.collision_vis_ori = self.collision_vis.clone()
        flag_no_need_local = torch.logical_or(self.collision_rigid, self.collision_vis == False)
        flag_no_need_local = torch.logical_or(flag_no_need_local, self.cur_episode_length == 0)
        env_ids = ~flag_no_need_local   # need local planner to turn collision_vis to False

        if torch.sum(env_ids) == 0:
            pass
        else:
            occ_maps_bin_cls = 1. - (self.occ_maps_tri_cls[env_ids] >= 0.).to(torch.float32)
            local_path_cuda = self.bfs_pathfinding(occ_maps_bin_cls, \
                                                   self.poses_idx_old[env_ids], self.poses_idx[env_ids])

            # if path exists, collision_vis: True -> False
            self.collision_vis[env_ids] = local_path_cuda == -1.  # NOTE: do not consider the length of local path
            # self.collision_vis[env_ids] = torch.logical_or(local_path_cuda == -1.,
            #                                                local_path_cuda >= self.grid_size // 2)  

        self.collision_flag = torch.logical_or(self.collision_rigid, self.collision_vis)

    def post_physics_step(self, if_reset=False):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)   # w/o using contact force

        self.episode_length_buf += 1

        obs, rewards, dones, infos = self.get_step_return()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return (obs, rewards, dones, infos) if not if_reset else obs

    def get_step_return(self):
        # render sensors and refresh camera tensors
        assert self.cfg.return_visual_observation, \
            "Images should be returned in this environment!"
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        self.post_process_main_camera_tensor()  # [num_env, num_cam, H, W]
        self.gym.end_access_image_tensors(self.sim)

        self.update_observation()
        self.check_motion_collision_local_2d()
        self.compute_reward()   # include check_termination()

        obs = {
            "state": self.ego_pose_buf.view(self.num_envs, -1),  # [num_env, pose_buffer_size * pose_size]
            "ego_map_2D": self.ego_prob_maps.reshape(self.num_envs, -1), # [num_env, H * W]
        }

        if self.visualize_flag:
            self.visualize_all_2d_occ_map_eval_local()
            # self.eval_all_envs_multi_round()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        rewards, dones, infos = self.rew_buf, self.reset_buf.clone(), self.extras
        self.update_extra_episode_info(rewards=rewards, dones=dones)
        self.reset_buf[env_ids] = 0

        return obs, rewards, dones, infos

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        num_env_ids = len(env_ids)
        device = self.device

        if num_env_ids == 0:
            return

        # reset active scenes with the probability of 1.0
        new_inactive_env_mask = torch.rand(len(env_ids), device=self.device) <= 1.

        if new_inactive_env_mask.sum() != 0:
            new_inactive_env_ids = env_ids[new_inactive_env_mask]       # active -> inactive
            new_inactive_env_ids_list = new_inactive_env_ids.tolist()   # env_idx to be moving
            num_new_inactive_env_ids = len(new_inactive_env_ids_list)

            new_inactive_scene_ids = [self.active_scene_ids[env_idx] for env_idx in new_inactive_env_ids_list]
            new_inactive_scene_ids_tensor = torch.tensor(new_inactive_scene_ids, dtype=torch.long, device=device)

            new_active_scene_ids = [(self.active_scene_ids[env_idx] + random.randint(1, self.scene_per_env - 1)) \
                                    % self.scene_per_env + env_idx * self.scene_per_env \
                                    for env_idx in new_inactive_env_ids_list]
            new_active_scene_ids_tensor = torch.tensor(new_active_scene_ids, dtype=torch.long, device=device)

            # Update active/inactive scene ids
            self.active_scene_ids = [self.active_scene_ids[env_idx] \
                                     if env_idx not in new_inactive_env_ids_list \
                                        else new_active_scene_ids.pop(0) \
                                     for env_idx in range(self.num_envs)]
            self.inactive_scene_ids = [scene_idx for scene_idx in range(self.num_scene) \
                                       if scene_idx not in self.active_scene_ids]

            # NOTE: move the scenes. compute the indices in the root state tensor
            new_inactive_scene_ids_root = new_inactive_scene_ids_tensor + \
                                            torch.div(new_inactive_scene_ids_tensor, self.scene_per_env).to(torch.long) + 1
            new_active_scene_ids_root = new_active_scene_ids_tensor + \
                                            torch.div(new_active_scene_ids_tensor, self.scene_per_env).to(torch.long) + 1
            
            self.root_states[new_inactive_scene_ids_root, :3] = self.inactive_xyz.repeat(num_new_inactive_env_ids, 1)
            self.root_states[new_active_scene_ids_root, :3] = self.env_origins[new_inactive_env_ids]

            # [num_scene] -> [num_env]
            self.env_to_scene = torch.tensor(self.active_scene_ids, device=self.device) # [num_env]

            self.range_gt_scenes[new_inactive_env_ids] = self.range_gt[new_active_scene_ids_tensor]
            self.voxel_size_gt_scenes[new_inactive_env_ids] = self.voxel_size_gt[new_active_scene_ids_tensor]
            self.num_valid_pixel_gt_scenes[new_inactive_env_ids] = self.num_valid_pixel_gt[new_active_scene_ids_tensor]
            self.layout_maps_height_scenes[new_inactive_env_ids] = self.layout_maps_height[new_active_scene_ids_tensor]

        # reset robot states
        self._reset_root_states(env_ids)

        # Reset pose buffers
        self.ego_pose_buf[env_ids] = torch.zeros(num_env_ids, self.buffer_size, self.pose_size, 
                                                 dtype=torch.float32, device=device)
        self.world_pose_buf[env_ids] = torch.zeros(num_env_ids, self.buffer_size, self.pose_size, 
                                                   dtype=torch.float32, device=device) 

        # Reset reward buffers
        for buf_idx in range(self.buffer_size):
            self.reward_layout_ratio_buf[buf_idx][env_ids] = torch.zeros(num_env_ids, dtype=torch.float32, device=device)

        # Reset actions
        self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action, 
                                            dtype=torch.int64, device=device).repeat(len(env_ids), 1)

        # Reset initial poses randomly
        for env_idx in env_ids:
            scene_idx = self.env_to_scene[env_idx]
            self.poses[env_idx, :2] = self.init_maps_list[scene_idx][random.randint(0,len(self.init_maps_list[scene_idx])-1)]
        self.poses[:, 2] = self.motion_height

        if self.visualize_flag:
            for env_idx in env_ids:
                self.local_paths[env_idx] = []

        # Reset maps
        self.prob_map[env_ids] = torch.zeros(num_env_ids, self.grid_size, self.grid_size,
                                             dtype=torch.float32, device=device)
        self.scanned_gt_map[env_ids] = torch.zeros(num_env_ids, self.grid_size, self.grid_size,
                                                   dtype=torch.float32, device=device)

        # Reset episode buffer
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Update extras and episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.    # terminate and create new episode

        # log additional curriculum info and usend timeout info to the algorithm
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def update_observation_space(self):
        """ update observation and action space
        """
        if not self.cfg.return_visual_observation:
            return

        # action space: discrete
        action_space_size = (self.clip_actions_up - self.clip_actions_low + 1).cpu().numpy()
        self.action_space = MultiDiscrete(nvec=torch.Size(action_space_size))

        # observation space
        x_max = self.range_gt[:, 0].max().item()
        x_min = self.range_gt[:, 1].min().item()
        y_max = self.range_gt[:, 2].max().item()
        y_min = self.range_gt[:, 3].min().item()

        # clip_pose_world_up = [x_max, y_max, self.motion_height, 0., 0., 2 * np.pi]
        clip_pose_world_up = [x_max, y_max, self.motion_height, 0., 0., 0.]
        clip_pose_world_low = [x_min, y_min, self.motion_height, 0., 0., 0.]

        pose_up_bound = np.tile(clip_pose_world_up, self.buffer_size).astype(np.float32)
        pose_low_bound = np.tile(clip_pose_world_low, self.buffer_size).astype(np.float32)

        self.observation_space = Dict(
            {
                "state": Box(low=pose_low_bound, high=pose_up_bound,
                             shape=(self.buffer_size*self.pose_size,), dtype=np.float32),
                "ego_map_2D": Box(low=-1., high=2., shape=(128*128, ), dtype=np.float32),
            }
        )
        pass

    def _reward_surface_coverage_2d(self):
        """ Reward for exploring the layout of the scene."""
        layout_coverage = self.scanned_gt_map.sum(dim=(1, 2)) / self.num_valid_pixel_gt_scenes
        self.reward_layout_ratio_buf.extend([layout_coverage.clone()])

        rew_coverage = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-2]
        rew_coverage[self.collision_flag] = 0.
        return rew_coverage

    def _reward_collision(self):
        """ Penalize collisions. """
        return self.collision_flag.to(torch.float32)

    def _reward_termination(self):
        """ Terminal reward for reaching the coverage goal."""
        return self.reset_buf * (self.reward_layout_ratio_buf[-1] > self.ratio_threshold_rew)  # terminate when reaching the coverage goal

