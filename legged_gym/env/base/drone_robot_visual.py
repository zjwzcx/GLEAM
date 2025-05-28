import math
import os
from collections import deque
import isaacgym
from PIL import Image as im
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
from isaacgym import *
from legged_gym import *
from legged_gym.env.base.drone_robot import DroneRobot
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
import torch
from gym.spaces import Dict, Box


class DroneVisualInputEnv(DroneRobot):
    """
    In this env, some basic functions are implemented for debugging visual input and test adding obstacle
    """
    DEPTH_SENSE_DIST = -8
    CAMERA_ANGLE = 15

    def __init__(self, *args, **kwargs):
        self.camera_handles = []
        self.asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(isaacgym.__file__)))), "assets")
        super(DroneVisualInputEnv, self).__init__(*args, **kwargs)
        self.update_observation_space()
        self.keyboard_press_event = {}

    def update_observation_space(self):
        if not self.cfg.return_visual_observation:
            return
        if self.cam_type == gymapi.IMAGE_COLOR:
            channel = 4
            image_shape = (
                self.cfg.visual_input.stack, self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width,
                channel
            )

        else:
            image_shape = (
                self.cfg.visual_input.stack, self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width
            )
        image_space = Box(low=0, high=255, shape=image_shape, dtype=np.float64)
        self.observation_space = Dict({"state": Box(low=-100, high=100, shape=(self.num_obs, )), "image": image_space})

    def _init_buffers(self):
        super(DroneVisualInputEnv, self)._init_buffers()
        # additional stat
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not self.cfg.return_visual_observation:
            return
        self.cam_tensors = []
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], self.cam_type)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.cam_tensors.append(torch_cam_tensor)
        if self.cam_type == gymapi.IMAGE_COLOR:
            channel = 4
            self.visual_obs_buf = torch.zeros(
                self.num_envs,
                self.cfg.visual_input.stack,
                self.cfg.visual_input.camera_height,
                self.cfg.visual_input.camera_width,
                channel,
                dtype=torch.float64,
                device=self.device,
                requires_grad=False
            )
        elif self.cam_type == gymapi.IMAGE_DEPTH:
            # Note: Depth image only contain one channel
            self.visual_obs_buf = torch.zeros(
                self.num_envs,
                self.cfg.visual_input.stack,
                self.cfg.visual_input.camera_height,
                self.cfg.visual_input.camera_width,
                dtype=torch.float64,
                device=self.device,
                requires_grad=False
            )
        else:
            raise ValueError("please indictae the channel of {} image".format(self.cam_type))

    def _draw_debug_vis(self):
        super(DroneVisualInputEnv, self)._draw_debug_vis()
        if not self.cfg.return_visual_observation:
            return
        # Add Camera position debug
        self.gym.clear_lines(self.viewer)
        axes_geom = gymutil.AxesGeometry(0.2)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 12, 12, sphere_pose, color=(1, 1, 0))
        for i in range(self.num_envs):
            camera_handle = self.camera_handles[i]
            pose = self.gym.get_camera_transform(self.sim, self.envs[i], camera_handle)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _create_envs(self):
        if self.cfg.return_visual_observation:
            assert self.cfg.env.num_envs <= 1024, \
                "Please set num_envs <= 1024, since more envs may make the GPU broken"
        # if any error, clean all isaac related things thoroughly and avoid cuda error
        super(DroneVisualInputEnv, self)._create_envs()
        self.add_camera_to_actors()

    def add_camera_to_actors(self):
        if not self.cfg.return_visual_observation:
            return
        camera_properties = gymapi.CameraProperties()

        camera_properties.width = self.cfg.visual_input.camera_width
        camera_properties.height = self.cfg.visual_input.camera_height
        camera_properties.enable_tensors = True
        for i in range(len(self.envs)):
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_offset = gymapi.Vec3(*self.cfg.visual_input.cam_pos)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.CAMERA_ANGLE))
            actor_handle = self.actor_handles[i]
            body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], actor_handle, 0)

            self.gym.attach_camera_to_body(
                cam_handle, self.envs[i], body_handle, gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_TRANSFORM
            )
            self.camera_handles.append(cam_handle)

    def _additional_create(self, env_handle, env_index):
        if env_handle not in self.additional_actors:
            self.additional_actors[env_index] = []
        if self.cfg.debug_add_ball:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset = self.gym.load_asset(self.sim, self.asset_root, "urdf/ball.urdf", asset_options)
            color = gymapi.Vec3(1, 0.8, 0.4)
            for x in range(1, 4):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x + self.env_origins[env_index][0], self.env_origins[env_index][1], 0.1)
                pose.r = gymapi.Quat(0, 0, 0, 1)
                ahandle = self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)
                self.gym.set_rigid_body_color(env_handle, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.additional_actors[env_index].append(ahandle)

    def step(self, actions):
        o, privileged_o, r, d, i = super(DroneVisualInputEnv, self).step(actions)

        # render sensors and refresh camera tensors
        if self.cfg.return_visual_observation:
            sim = self.sim
            self.gym.step_graphics(sim)
            self.gym.render_all_camera_sensors(sim)
            self.gym.start_access_image_tensors(sim)
            self.post_process_camera_tensor()
            o = {"state": o, "image": self.visual_obs_buf}
            self.debug_save_image(o)
            self.gym.end_access_image_tensors(sim)
        self.update_extra_episode_info(rewards=r, dones=d)
        if self.cfg.return_privileged_observation:
            return {"obs": o, "privileged_obs": privileged_o}, r, d, i
        else:
            return o, r, d, i

    def debug_save_image(self, o):
        if self.cfg.debug_save_image_tensor and self.cfg.return_visual_observation:
            for i in range(self.num_envs):
                # write tensor to image
                fname = "cam-frame%d-env%d.png" % (self.common_step_counter, i)
                cam_img = o["image"][i][-1]
                if self.cfg.visual_input.normalization:
                    cam_img *= 255
                cam_img = cam_img.cpu().detach().numpy()
                image = im.fromarray(
                    cam_img.astype(np.uint8), mode="RGBA" if self.cam_type == gymapi.IMAGE_COLOR else "L"
                )
                image.save(fname)

    def update_extra_episode_info(self, rewards, dones):
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().detach().numpy().tolist())
        self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().detach().numpy().tolist())

        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0

        # self.extras["episode"]["episode_reward"] = np.mean(self.rewbuffer) if len(self.rewbuffer) > 0 else 0.
        self.extras["episode"]["episode_reward"] = np.mean(self.rewbuffer) / self.max_episode_length_s if len(self.rewbuffer) > 0 else 0.   # rollout/episode_reward
        self.extras["episode"]["episode_length"] = np.mean(self.lenbuffer) if len(self.lenbuffer) > 0 else 0.


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def reset_idx(self, env_ids):
        return super(DroneVisualInputEnv, self).reset_idx(env_ids)

    def add_visual_obs_noise(self):
        """Consider to add noise for the image observations"""
        pass

    def stack_visual_input_tensor(self, x):
        """
        This function is used for concating frames along with the time axis
        """
        self.visual_obs_buf = torch.cat((self.visual_obs_buf[:, 1:, ...], torch.unsqueeze(x, axis=1)), axis=1)

    def push_robots_y_axis(self):
        self.root_states[::self.skip, 7:9] = torch_rand_float(
            0, 10, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def push_robots_x_axis(self):
        self.root_states[::self.skip, 7:9] = torch_rand_float(
            10, 0, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def set_robots_roll_pitch_yaw(self, roll=0., pitch=0., yaw=0.):
        quat = quat_from_euler_xyz(roll * torch.ones((1, )), pitch * torch.ones((1, )), yaw * torch.ones((1, )))
        self.root_states[::self.skip, 3:7] = quat
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def set_robots_velocity(self, x_vel=0, y_vel=0, z_vel=0, local_coordinates=False):
        velocity = torch.Tensor([[x_vel, y_vel, z_vel] for _ in range(self.num_envs)]).to(self.device)
        if local_coordinates:
            velocity = quat_rotate(self.base_quat, velocity)
        self.root_states[::self.skip, 7:10] = velocity
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        if self.cam_type == gymapi.IMAGE_COLOR:
            if self.cfg.visual_input.normalization:
                new_images = new_images / 255
        elif self.cam_type == gymapi.IMAGE_DEPTH:
            new_images = torch.nan_to_num(new_images, neginf=0)
            new_images = torch.clamp(new_images, min=self.DEPTH_SENSE_DIST)
            new_images = 1 + (new_images / torch.min(new_images + 1e-4))
            if not self.cfg.visual_input.normalization:
                new_images = new_images * 255
        else:
            raise ValueError("Currently the {} is not fully tested".format(self.cam_type))
        self.stack_visual_input_tensor(new_images)

    @property
    def cam_type(self):
        return self.cfg.visual_input.type

    def render(self, sync_frame_time=True):
        if self.cfg.camera.first_view_camera and self.viewer is not None:
            position = self.base_pos[self.cfg.camera.env_to_track]
            r, p, y = isaacgym.torch_utils.get_euler_xyz(self.base_quat)
            camera_x = -torch.cos(y) * self.cfg.camera.dist + position[0]
            camera_y = -torch.sin(y) * self.cfg.camera.dist + position[1]
            height = self.cfg.camera.height + position[-1]
            self.set_camera(
                (camera_x[self.cfg.camera.env_to_track], camera_y[self.cfg.camera.env_to_track], height), (position)
            )
        events = {i.action: i.value > 0 for i in self.gym.query_viewer_action_events(self.viewer)}
        for evt in self.keyboard_press_event.keys():
            if evt in events and events[evt]:
                self.keyboard_press_event[evt] = True
            else:
                self.keyboard_press_event[evt] = False
        return super(DroneVisualInputEnv, self).render()

    def track_keypress_event(self, key, event_name):
        self.keyboard_press_event[event_name] = False
        self.gym.subscribe_viewer_keyboard_event(self.viewer, key, event_name)

    def has_event(self, event):
        if event not in self.keyboard_press_event:
            print("Not event: {}, Please track it before calling this func".format(event))
            return False
        return self.keyboard_press_event[event]


if __name__ == "__main__":
    import legged_complex_env

    args = get_args()
    args.headless = False
    args.sim_device = "cuda"
    env_cfg, _ = task_registry.get_cfgs(name="legged_visual_input")
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.camera.first_view_camera = True
    env, _ = task_registry.make_env("legged_visual_input", args, env_cfg)
    env.debug_viz = True
    env.reset()
    while True:
        o, r, d, i = env.step(env.sample_actions())
        obs = {"state": o["state"][0].cpu().detach().numpy(), "image": o["image"][0].cpu().detach().numpy()}
        assert env.observation_space.contains(obs), "OBS shape errpr!!"
