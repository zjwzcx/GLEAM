# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import OPEN_ROBOT_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs:
            runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        #env parameters
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        try:
            env_cfg.env.num_observations = args.num_observations
        except:
            pass
        try:
            env_cfg.env.num_privileged_obs = args.num_privileged_obs
        except:
            pass
        try:
            env_cfg.env.num_actions = args.num_actions
        except:
            pass
        try:
            env_cfg.env.env_spacing = args.env_spacing
        except:
            pass
        try:
            env_cfg.env.send_timeouts = args.send_timeouts
        except:
            pass
        try:
            env_cfg.env.episode_length_s = args.episode_length_s
        except:
            pass
        try:
            env_cfg.env.unflatten_terrain = args.unflatten_terrain
        except:
            pass
        #terrain parameters
        try:
            env_cfg.terrain.mesh_type = args.mesh_type
        except:
            pass
        try:
            env_cfg.terrain.horizontal_scale = args.horizontal_scale
        except:
            pass
        try:
            env_cfg.terrain.vertical_scale = args.vertical_scale
        except:
            pass
        try:
            env_cfg.terrain.border_size = args.border_size
        except:
            pass
        try:
            env_cfg.terrain.curriculum = args.curriculum
        except:
            pass
        try:
            env_cfg.terrain.static_friction = args.static_friction
        except:
            pass
        try:
            env_cfg.terrain.dynamic_friction = args.dynamic_friction
        except:
            pass
        try:
            env_cfg.terrain.restitution = args.restitution
        except:
            pass
        # rough terrain only:
        try:
            env_cfg.terrain.measure_heights = args.measure_heights
        except:
            pass
        try:
            env_cfg.terrain.measured_points_x = args.measured_points_x
        except:
            pass
        try:
            env_cfg.terrain.measured_points_y = args.measured_points_y
        except:
            pass
        try:
            env_cfg.terrain.selected = args.selected
        except:
            pass
        try:
            env_cfg.terrain.terrain_kwargs = args.terrain_kwargs
        except:
            pass
        try:
            env_cfg.terrain.max_init_terrain_level = args.max_init_terrain_level
        except:
            pass
        try:
            env_cfg.terrain.terrain_length = args.terrain_length
        except:
            pass
        try:
            env_cfg.terrain.terrain_width = args.terrain_width
        except:
            pass
        try:
            env_cfg.terrain.num_rows = args.num_rows
        except:
            pass
        try:
            env_cfg.terrain.num_cols = args.num_cols
        except:
            pass
        try:
            env_cfg.terrain.terrain_proportions = args.terrain_proportions
        except:
            pass
        # trimesh only:
        try:
            env_cfg.terrain.slope_treshold = args.slope_treshold
        except:
            pass
        #commands
        try:
            env_cfg.commands.curriculum = args.curriculum
        except:
            pass
        try:
            env_cfg.commands.max_curriculum = args.max_curriculum
        except:
            pass
        try:
            env_cfg.commands.num_commands = args.num_commands
        except:
            pass
        try:
            env_cfg.commands.resampling_time = args.resampling_time
        except:
            pass
        try:
            env_cfg.commands.heading_command = args.heading_command
        except:
            pass
        try:
            env_cfg.commands.ranges.lin_vel_x = args.lin_vel_x
        except:
            pass
        try:
            env_cfg.commands.ranges.lin_vel_y = args.lin_vel_y
        except:
            pass
        try:
            env_cfg.commands.ranges.ang_vel_yaw = args.ang_vel_yaw
        except:
            pass
        try:
            env_cfg.commands.ranges.heading = args.heading
        except:
            pass
        #init_state
        try:
            env_cfg.init_state.pos = args.pos
        except:
            pass
        try:
            env_cfg.init_state.rot = args.rot
        except:
            pass
        try:
            env_cfg.init_state.lin_vel = args.lin_vel
        except:
            pass
        try:
            env_cfg.init_state.ang_vel = args.ang_vel
        except:
            pass
        try:
            env_cfg.init_state.default_joint_angles = args.default_joint_angles
        except:
            pass
        #control
        try:
            env_cfg.control.control_type = args.control_type
        except:
            pass
        try:
            env_cfg.control.stiffness = args.stiffness
        except:
            pass
        try:
            env_cfg.control.damping = args.damping
        except:
            pass
        try:
            env_cfg.control.action_scale = args.action_scale
        except:
            pass
        try:
            env_cfg.control.decimation = args.decimation
        except:
            pass
        #asset
        try:
            env_cfg.asset.file = args.file
        except:
            pass
        try:
            env_cfg.asset.name = args.name
        except:
            pass
        try:
            env_cfg.asset.foot_name = args.foot_name
        except:
            pass
        try:
            env_cfg.asset.penalize_contacts_on = args.penalize_contacts_on
        except:
            pass
        try:
            env_cfg.asset.terminate_after_contacts_on = args.terminate_after_contacts_on
        except:
            pass
        try:
            env_cfg.asset.disable_gravity = args.disable_gravity
        except:
            pass
        try:
            env_cfg.asset.collapse_fixed_joints = args.collapse_fixed_joints
        except:
            pass
        try:
            env_cfg.asset.fix_base_link = args.fix_base_link
        except:
            pass
        try:
            env_cfg.asset.default_dof_drive_mode = args.default_dof_drive_mode
        except:
            pass
        try:
            env_cfg.asset.self_collisions = args.self_collisions
        except:
            pass
        try:
            env_cfg.asset.replace_cylinder_with_capsule = args.replace_cylinder_with_capsule
        except:
            pass
        try:
            env_cfg.asset.flip_visual_attachments = args.flip_visual_attachments
        except:
            pass
        try:
            env_cfg.asset.density = args.density
        except:
            pass
        try:
            env_cfg.asset.angular_damping = args.angular_damping
        except:
            pass
        try:
            env_cfg.asset.linear_damping = args.linear_damping
        except:
            pass
        try:
            env_cfg.asset.max_angular_velocity = args.max_angular_velocity
        except:
            pass
        try:
            env_cfg.asset.max_linear_velocity = args.max_linear_velocity
        except:
            pass
        try:
            env_cfg.asset.armature = args.armature
        except:
            pass
        try:
            env_cfg.asset.thickness = args.thickness
        except:
            pass
        #domain rand
        try:
            env_cfg.domain_rand.randomize_friction = args.randomize_friction
        except:
            pass
        try:
            env_cfg.domain_rand.friction_range = args.friction_range
        except:
            pass
        try:
            env_cfg.domain_rand.randomize_base_mass = args.randomize_base_mass
        except:
            pass
        try:
            env_cfg.domain_rand.added_mass_range = args.added_mass_range
        except:
            pass
        try:
            env_cfg.domain_rand.push_robots = args.push_robots
        except:
            pass
        try:
            env_cfg.domain_rand.push_interval_s = args.push_interval_s
        except:
            pass
        try:
            env_cfg.domain_rand.max_push_vel_xy = args.max_push_vel_xy
        except:
            pass
        #rewards
        try:
            env_cfg.rewards.scales.lin_vel_z = args.lin_vel_z
        except:
            pass
        try:
            env_cfg.rewards.scales.ang_vel_xy = args.ang_vel_xy
        except:
            pass
        try:
            env_cfg.rewards.scales.orientation = args.orientation
        except:
            pass
        try:
            env_cfg.rewards.scales.base_height = args.base_height
        except:
            pass
        try:
            env_cfg.rewards.scales.torques = args.torques
        except:
            pass
        try:
            env_cfg.rewards.scales.dof_vel = args.dof_vel
        except:
            pass
        try:
            env_cfg.rewards.scales.dof_acc = args.dof_acc
        except:
            pass
        try:
            env_cfg.rewards.scales.action_rate = args.action_rate
        except:
            pass
        try:
            env_cfg.rewards.scales.collision = args.collision
        except:
            pass
        try:
            env_cfg.rewards.scales.termination = args.termination
        except:
            pass
        try:
            env_cfg.rewards.scales.dof_pos_limits = args.dof_pos_limits
        except:
            pass
        try:
            env_cfg.rewards.scales.dof_vel_limits = args.dof_vel_limits
        except:
            pass
        try:
            env_cfg.rewards.scales.torque_limits = args.torque_limits
        except:
            pass
        try:
            env_cfg.rewards.scales.tracking_lin_vel = args.tracking_lin_vel
        except:
            pass
        try:
            env_cfg.rewards.scales.x_afap = args.x_afap
        except:
            pass
        try:
            env_cfg.rewards.scales.tracking_x_vel = args.tracking_x_vel
        except:
            pass
        try:
            env_cfg.rewards.scales.tracking_ang_vel = args.tracking_ang_vel
        except:
            pass
        try:
            env_cfg.rewards.scales.feet_air_time = args.feet_air_time
        except:
            pass
        try:
            env_cfg.rewards.scales.stumble = args.stumble
        except:
            pass
        try:
            env_cfg.rewards.scales.stand_still = args.stand_still
        except:
            pass
        try:
            env_cfg.rewards.scales.feet_contact_forces = args.feet_contact_forces
        except:
            pass

        try:
            env_cfg.rewards.only_positive_rewards = args.only_positive_rewards
        except:
            pass
        try:
            env_cfg.rewards.tracking_sigma = args.tracking_sigma
        except:
            pass
        try:
            env_cfg.rewards.soft_dof_pos_limit = args.soft_dof_pos_limit
        except:
            pass
        try:
            env_cfg.rewards.soft_dof_vel_limit = args.soft_dof_vel_limit
        except:
            pass
        try:
            env_cfg.rewards.soft_torque_limit = args.soft_torque_limit
        except:
            pass
        try:
            env_cfg.rewards.base_height_target = args.base_height_target
        except:
            pass
        try:
            env_cfg.rewards.max_contact_force = args.max_contact_force
        except:
            pass
        try:
            env_cfg.rewards.scales.forward = args.forward
        except:
            pass
        try:
            env_cfg.rewards.scales.lateral_movement_and_rotations = args.lateral_movement_and_rotations
        except:
            pass
        try:
            env_cfg.rewards.scales.work = args.work
        except:
            pass
        try:
            env_cfg.rewards.scales.ground_impact = args.ground_impact
        except:
            pass
        try:
            env_cfg.rewards.scales.smoothness = args.smoothness
        except:
            pass
        try:
            env_cfg.rewards.scales.action_magnitude = args.action_magnitude
        except:
            pass
        try:
            env_cfg.rewards.scales.joint_speed = args.joint_speed
        except:
            pass
        try:
            env_cfg.rewards.scales.z_acc = args.z_acc
        except:
            pass
        try:
            env_cfg.rewards.scales.foot_slip = args.foot_slip
        except:
            pass
        try:
            env_cfg.rewards.scales.energy = args.energy
        except:
            pass
        try:
            env_cfg.rewards.scales.alive = args.alive
        except:
            pass
        #normalization

    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.exp_name is not None:
            cfg_train.runner.run_name = args.exp_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args(additional_args=None):
    additional_args = additional_args or []
    if additional_args is not None:
        assert isinstance(additional_args, list), "Additional args should be included in list"
        for arg in additional_args:
            assert isinstance(arg, dict), "args should be in dict type"
            assert "name" in arg, "include a name in args!"
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint"
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided."
        },
        {
            "name": "--exp_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided."
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times"
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training"
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided."
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
            "help": "Random seed. Overrides config file if provided."
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided."
        },
        {
            "name": "--stop_wandb",
            "action": "store_true",
            "default": False,
            "help": "Use wandb to log or not"
        },
        {
            "name": "--mode",
            "type": int,
        },
    ] + additional_args
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
