from isaacgym import gymapi
from isaacgym.torch_utils import *
from legged_gym import OPEN_ROBOT_ROOT_DIR
import os
import torch
import numpy as np
from gleam.env.env_gleam_base import Env_GLEAM_Base
from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1


class Env_GLEAM_Stage2(Env_GLEAM_Stage1):
    def __init__(self, *args, **kwargs):
        """
        Training set of stage 2 includes 512 indoor scenes from:
            procthor_2-bed-2-bath_256,
            procthor_7-room-3-bed_96, 
            procthor_12-room_64,
            gibson_96.
        """
        self.visualize_flag = False     # training
        # self.visualize_flag = True    # visualization

        # self.num_scene = 4 # debug
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
        dataset_name = "stage2_512"
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
        dataset_name = "stage2_512"
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
        super()._init_buffers()

        # visualization
        if self.visualize_flag:
            # visualize scenes
            self.vis_obj_idx = 0
            print("Visualization object index: ", self.vis_obj_idx)

            self.reset_once_flag = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.reset_once_cr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.local_paths = [[] for _ in range(self.num_envs)]

            self.save_path = f'./gleam/scripts/video/train_stage2_512'
            os.makedirs(self.save_path, exist_ok=True)
