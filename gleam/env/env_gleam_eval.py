from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym import OPEN_ROBOT_ROOT_DIR
import os
import torch
import random
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from collections import deque
from gleam.utils.utils import create_e2w_from_poses
from gleam.env.env_gleam_base import Env_GLEAM_Base
from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1


class Env_GLEAM_Eval(Env_GLEAM_Stage1):
    def __init__(self, *args, **kwargs):
        """
        Test set includes 128 indoor scenes from:
            procthor_4-room_28_eval,
            procthor_2-bed-2-bath_24_eval,
            procthor_7-room-3-bed_18_eval,
            procthor_12-room-3-bed_6_eval,
            hssd_10_eval,
            gibson_24_eval,
            mp3d_18_eval
        """
        self.visualize_flag = False     # evaluation
        # self.visualize_flag = True    # visualization

        # self.num_scene = 4    # debug
        # print("*"*50, "num_scene: ", self.num_scene, "*"*50)

        self.num_scene = 128

        super(Env_GLEAM_Base, self).__init__(*args, **kwargs)

    def _additional_create(self, env_handle, env_index):
        """
        If there are N training scenes and M environments, each environment will load N//M scenes.
        Only the first scene (idx == 0) is active, and the others are inactive.
        """
        assert self.cfg.return_visual_observation, "visual observation should be returned!"
        assert self.num_scene >= self.num_envs, "num_scene should be larger than num_envs"

        # urdf load, create actor
        dataset_name = "eval_128"
        urdf_path = f"data_gleam/{dataset_name}/urdf"

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
        dataset_name = "eval_128"
        gt_path = os.path.join(OPEN_ROBOT_ROOT_DIR, f"data_gleam/{dataset_name}/gt/")

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

        # list of []
        self.layout_pc = [(torch.nonzero(self.layout_maps_height[idx]) / (self.grid_size - 1) * 2 - 1) * self.range_gt[idx, :4:2]
                          for idx in range(self.num_scene)]

        # [num_scene]
        self.num_valid_pixel_gt = self.layout_maps_height.sum(dim=(1, 2))

        # [num_scene, 128, 128]
        init_maps = torch.load(gt_path+f"{dataset_name}_{self.grid_size}_init_map_1d5.pt", 
                                map_location=self.device)[:self.num_scene]
        init_maps /= 255.

        self.init_maps_list_eval = torch.load("gleam/test/eval_128_init_10.pt")

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
        self.vis_obj_idx = -1
        print("Visualization object index: ", self.vis_obj_idx)

        self.reset_once_flag = torch.zeros(self.num_scene, dtype=torch.float32, device=self.device)
        self.reset_once_cr = torch.zeros(self.num_scene, dtype=torch.float32, device=self.device)

        # evaluate on all envs
        self.num_eval_round = 10
        self.reset_num_count_round = torch.zeros(self.num_scene, dtype=torch.float32, device=self.device)    # count the number of finished rounds
        self.reset_multi_round_cr = torch.zeros(self.num_scene, self.num_eval_round, dtype=torch.float32, device=self.device)
        self.reset_multi_round_AUC = torch.zeros(self.num_scene, self.num_eval_round, 50, dtype=torch.float32, device=self.device)
        self.reset_multi_round_chamfer_dist = torch.zeros(self.num_scene, self.num_eval_round, dtype=torch.float32, device=self.device)
        self.reset_multi_round_traj_length = torch.zeros(self.num_scene, self.num_eval_round, dtype=torch.float32, device=self.device)
        self.reset_multi_round_path_length = torch.zeros(self.num_scene, self.num_eval_round, dtype=torch.float32, device=self.device)

        self.local_paths = [[] for _ in range(self.num_envs)]
        self.scanned_pc_coord = [[] for _ in range(self.num_envs)]

        self.save_path = f'./gleam/scripts/video/eval_gleam_128'
        os.makedirs(self.save_path, exist_ok=True)

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

        # evaluate trajectory length
        scene_indices = self.env_to_scene
        round_indices = self.reset_num_count_round[scene_indices].long()
        mask_unfinished = (round_indices < self.num_eval_round)
        scene_indices = scene_indices[mask_unfinished]
        round_indices = round_indices[mask_unfinished]

        move_dists = torch.norm(self.move_dist, dim=1)
        self.reset_multi_round_traj_length.index_put_(
            (scene_indices, round_indices),
            move_dists[mask_unfinished],
            accumulate=True
        )

        current_move_xyz = torch.cat([self.move_dist, 
                                      torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)], dim=1) # [num_env, 3]
        self.set_collision_cam_pose(current_poses, current_move_xyz)  # set the pose of collision camera

        self.render()
        self.set_state(self.poses)
        obs, rewards, dones, infos = self.post_physics_step()
        return obs, rewards, dones, infos

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

        # # meaningless wander
        # recent_CR_thres = 0.01

        # recent_CR = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-self.recent_num]  # [num_envs]
        # meaningless_wander = (recent_CR < recent_CR_thres)  # [num_envs]
        # meaningless_wander *= (self.cur_episode_length > self.recent_num)
        # self.reset_buf |= meaningless_wander.clone()

        # max_step
        # self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.termination.max_step_done:  # <-
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)    # [num_envs]
            self.reset_buf |= self.time_out_buf

        # coverage ratio > threshold
        last_ratio = self.reward_layout_ratio_buf[-1]
        self.reset_buf |= (last_ratio > self.ratio_threshold_term)

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


        self.eval_all_envs_multi_round()

        if self.visualize_flag:
            self.visualize_all_2d_occ_map_eval_local()

        env_ids = self.reset_buf.nonzero().flatten()    # [num_env_ids], to be reset
        env_ids_reset = self.arange_envs[self.reset_num_count_round[self.env_to_scene] == self.num_eval_round]
        env_ids = torch.cat([env_ids, env_ids_reset], dim=0)     # [num_env_ids]
        env_ids = torch.unique(env_ids, sorted=False)
        for env_idx in env_ids:
            scene_idx = self.env_to_scene[env_idx]
            round_idx = int(self.reset_num_count_round[scene_idx].item())
            if round_idx == self.num_eval_round and scene_idx not in self.active_scene_ids:
                continue

            if self.reset_multi_round_chamfer_dist[scene_idx, round_idx-1] == 0.:
                scanned_pc_coord = torch.cat(self.scanned_pc_coord[env_idx], dim=0).unsqueeze(0)   # [N=1, num_point, 2]
                scanned_kd_tree = KDTree(scanned_pc_coord[0].cpu().numpy())

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(torch.cat([self.layout_pc[scene_idx].cpu(), torch.zeros(self.layout_pc[scene_idx].shape[0], 1)], dim=1).numpy())
                # o3d.io.write_point_cloud(f"{self.save_path}/scene_{scene_idx}_round_{round_idx}_gt.pcd", pcd)

                # pcd_scanned = o3d.geometry.PointCloud()
                # pcd_scanned.points = o3d.utility.Vector3dVector(torch.cat([scanned_pc_coord[0].cpu(), torch.zeros(scanned_pc_coord[0].shape[0], 1)], dim=1).numpy())
                # o3d.io.write_point_cloud(f"{self.save_path}/scene_{scene_idx}_round_{round_idx}_scanned.pcd", pcd_scanned)

                # chamfer distance (unit: cm)
                distance, _ = scanned_kd_tree.query(self.layout_pc[scene_idx].cpu().numpy())
                self.reset_multi_round_chamfer_dist[scene_idx, round_idx-1] = 100 * np.mean(distance)

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

        scene_ids = self.env_to_scene[env_ids]
        new_inactive_env_mask = self.reset_num_count_round[scene_ids] == self.num_eval_round
        

        if new_inactive_env_mask.sum() != 0:
            new_inactive_env_ids = env_ids[new_inactive_env_mask]       # active -> inactive
            new_inactive_env_ids_list = new_inactive_env_ids.tolist()   # env_idx to be moving
            num_new_inactive_env_ids = len(new_inactive_env_ids_list)

            new_inactive_scene_ids = [self.active_scene_ids[new_inactive_env_idx] for new_inactive_env_idx in new_inactive_env_ids_list]
            new_inactive_scene_ids_tensor = torch.tensor(new_inactive_scene_ids, dtype=torch.long, device=device)

            new_active_scene_ids = [(self.active_scene_ids[new_inactive_env_idx] + 1) % self.scene_per_env + \
                                        new_inactive_env_idx * self.scene_per_env \
                                    for new_inactive_env_idx in new_inactive_env_ids_list]
            new_active_scene_ids_tensor = torch.tensor(new_active_scene_ids, dtype=torch.long, device=device)

            # Update active/inactive scene ids
            # if original elements is not moved this round, then keep it in the active_scene_ids; If so, replace it with new_active_scene_ids
            self.active_scene_ids = [self.active_scene_ids[env_idx] \
                                     if env_idx not in new_inactive_env_ids_list \
                                        else new_active_scene_ids.pop(0) \
                                     for env_idx in range(self.num_envs)]
            self.inactive_scene_ids = [scene_idx for scene_idx in range(self.num_scene) \
                                       if (self.reset_num_count_round[scene_idx] != self.num_eval_round) and (scene_idx not in self.active_scene_ids)]

            # NOTE: move the scenes. compute the indices in the root state tensor
            new_inactive_scene_ids_root = new_inactive_scene_ids_tensor + torch.div(new_inactive_scene_ids_tensor, self.scene_per_env).to(torch.long) + 1
            self.root_states[new_inactive_scene_ids_root, :3] = self.inactive_xyz.repeat(num_new_inactive_env_ids, 1)

            if new_active_scene_ids_tensor.shape[0] != 0:
                new_active_scene_ids_root = new_active_scene_ids_tensor + torch.div(new_active_scene_ids_tensor, self.scene_per_env).to(torch.long) + 1
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

        # Reset scanned point cloud
        for env_idx in env_ids:
            self.scanned_pc_coord[env_idx] = []

        # Reset actions
        self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action, 
                                            dtype=torch.int64, device=device).repeat(len(env_ids), 1)

        # Reset initial poses randomly
        for env_idx in env_ids:
            scene_idx = self.env_to_scene[env_idx]
            round_idx = int(self.reset_num_count_round[scene_idx].item())
            if round_idx == self.num_eval_round:
                continue
            self.poses[env_idx, :2] = self.init_maps_list_eval[scene_idx][round_idx]
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

    def eval_all_envs_multi_round(self):
        """ Evaluate on all envs + average CR over 10 rounds. """
        save_path_occ = os.path.join(self.save_path, "all_2D_occ_map_local")
        os.makedirs(save_path_occ, exist_ok=True)

        for env_idx in range(self.num_envs):
            scene_idx = self.env_to_scene[env_idx]
            round_idx = int(self.reset_num_count_round[scene_idx].item())
            if scene_idx not in self.active_scene_ids or self.reset_num_count_round[scene_idx] == self.num_eval_round or \
                self.cur_episode_length[env_idx] == 0 or self.reset_buf[env_idx] == 0 or round_idx >= self.num_eval_round:
                continue

            self.reset_multi_round_cr[scene_idx, round_idx] = self.reward_layout_ratio_buf[-1][env_idx]
            self.reset_multi_round_path_length[scene_idx, round_idx] = self.episode_length_buf[env_idx]
            self.reset_num_count_round[scene_idx] += 1

        process = self.reset_num_count_round.sum()
        with tqdm(total=self.num_scene * self.num_eval_round, desc="Evaluating all environments") as pbar:
            pbar.update(process)

        if process == self.num_scene * self.num_eval_round:
            torch.save(self.reset_multi_round_cr, os.path.join(self.save_path, "reset_multi_round_cr.pt"))  # [num_env, num_round]
            torch.save(self.reset_multi_round_AUC, os.path.join(self.save_path, "reset_multi_round_AUC.pt"))    # [num_env, num_round, max_episode_length=50]
            torch.save(self.reset_multi_round_chamfer_dist, os.path.join(self.save_path, "reset_multi_round_chamfer_dist.pt"))  # [num_env, num_round]
            torch.save(self.reset_multi_round_traj_length, os.path.join(self.save_path, "reset_multi_round_traj_length.pt"))    # [num_env, num_round]
            torch.save(self.reset_multi_round_path_length, os.path.join(self.save_path, "reset_multi_round_path_length.pt"))    # [num_env, num_round]


            # print("All CR: ", self.reset_multi_round_cr)
            print("*"*50)
            mean_auc = torch.zeros(self.num_scene, self.num_eval_round, dtype=torch.float32, device=self.device)
            for step_idx in range(self.max_episode_length):
                mean_auc += self.reset_multi_round_AUC[:, :, step_idx] * (self.max_episode_length - step_idx)
            mean_auc /= self.max_episode_length
            print("[CR] Average ProcTHOR & HSSD: ", self.reset_multi_round_cr[:86].mean(dim=(0,1)))
            print("[CR] Average Gibson & MP3D: ", self.reset_multi_round_cr[86:].mean(dim=(0,1)))
            print("[CR] Average All: ", self.reset_multi_round_cr.mean(dim=(0,1)))
            print("*"*50)
            print("[AUC] Average ProcTHOR & HSSD: ", mean_auc[:86].mean(dim=(0,1)))
            print("[AUC] Average Gibson & MP3D: ", mean_auc[86:].mean(dim=(0,1)))
            print("[AUC] Average All: ", mean_auc.mean(dim=(0,1)))
            print("*"*50)
            print("[Comp.] Average ProcTHOR & HSSD: ", self.reset_multi_round_chamfer_dist[:86].mean(dim=(0,1)))
            print("[Comp.] Average Gibson & MP3D: ", self.reset_multi_round_chamfer_dist[86:].mean(dim=(0,1)))
            print("[Comp.] Average All: ", self.reset_multi_round_chamfer_dist.mean(dim=(0,1)))
            print("*"*50)
            print("[Path] Average ProcTHOR & HSSD: ", self.reset_multi_round_path_length[:86].mean(dim=(0,1)))
            print("[Path] Average Gibson & MP3D: ", self.reset_multi_round_path_length[86:].mean(dim=(0,1)))
            print("[Path] Average All: ", self.reset_multi_round_path_length.mean(dim=(0,1)))
            print("*"*50)
            print("[Trajectory] Average ProcTHOR & HSSD: ", self.reset_multi_round_traj_length[:86].mean(dim=(0,1)))
            print("[Trajectory] Average Gibson & MP3D: ", self.reset_multi_round_traj_length[86:].mean(dim=(0,1)))
            print("[Trajectory] Average All: ", self.reset_multi_round_traj_length.mean(dim=(0,1)))
            print("*"*50)
            non_one_step = self.reset_multi_round_traj_length != 0
            print("[Cov./Traj.] Average ProcTHOR & HSSD: ", (self.reset_multi_round_cr[:86][non_one_step[:86]] / self.reset_multi_round_traj_length[:86][non_one_step[:86]]).mean())
            print("[Cov./Traj.] Average Gibson & MP3D: ", (self.reset_multi_round_cr[86:][non_one_step[86:]] / self.reset_multi_round_traj_length[86:][non_one_step[86:]]).mean())
            print("[Cov./Traj.] Average All: ", (self.reset_multi_round_cr[non_one_step] / self.reset_multi_round_traj_length[non_one_step]).mean())
            print("Done")
            exit()

    def _reward_surface_coverage_2d(self):
        """ Reward for exploring the layout of the scene."""
        layout_coverage = self.scanned_gt_map.sum(dim=(1, 2)) / self.num_valid_pixel_gt_scenes
        self.reward_layout_ratio_buf.extend([layout_coverage.clone()])

        rew_coverage = self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-2]
        rew_coverage[self.collision_flag] = 0.

        env_ids = torch.arange(self.num_envs, device=self.device)
        scene_ids = self.env_to_scene[env_ids]
        round_ids = self.reset_num_count_round[scene_ids].long()
        mask_round = round_ids < self.num_eval_round

        scene_ids = scene_ids[mask_round]
        env_ids = env_ids[mask_round]
        round_ids = round_ids[mask_round]
        self.reset_multi_round_AUC[scene_ids, round_ids, self.cur_episode_length[env_ids].long()] = rew_coverage[mask_round].clone()

        return rew_coverage
