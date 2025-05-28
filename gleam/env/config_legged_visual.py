import torch

from legged_gym.env.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from isaacgym import gymapi


class LeggedVisualInputConfig(A1RoughCfg):
    debug_viz = False
    debug_add_ball = False
    debug_save_image_tensor = False
    return_visual_observation = True
    return_privileged_observation = False

    class camera:
        first_view_camera = False
        env_to_track = 0
        dist = 1.5
        height = 1

    class env(A1RoughCfg.env):
        num_observations = 48
        episode_length_s = 20  # in second !!!!!

    class terrain(A1RoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        # restitution = 1

    class visual_input:
        camera_width = 320
        camera_height = 240
        type = gymapi.IMAGE_COLOR
        stack = 5  # consecutive frames to stack
        normalization = True
        cam_pos = (0.27, 0, 0.)


class LeggedVIsualInputCfgPPO(A1RoughCfgPPO):
    class algorithm(A1RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(A1RoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'legged_visual_input'


class LeggedStaticConfig(LeggedVisualInputConfig):
    class visual_input(LeggedVisualInputConfig.visual_input):
        camera_width = 64
        camera_height = 64
        type = gymapi.IMAGE_DEPTH
        stack = 4  # consecutive frames to stack
        normalization = True

    class env(LeggedVisualInputConfig.env):
        episode_length_s = 40  # episode length in seconds
        unflatten_terrain = True

    class commands:
        # only move forward
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 20.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [.4, .8]  # min max [m/s]
            lin_vel_y = [-.0, .0]  # min max [m/s]
            ang_vel_yaw = [-.0, .0]  # min max [rad/s]
            heading = [-.0, .0]

    class rewards:
        # copied from the legged robot config
        class scales:
            feet_stumble = -0.0

            lin_vel_z = -2.0
            # Penalize z axis base linear velocity
            ang_vel_xy = .0
            # Penalize xy axes base angular velocity
            orientation = -0.5
            # Penalize non flat base orientation
            base_height = -1
            # Penalize base height away from target
            torques = -0.0002
            # Penalize torques
            dof_vel = -0.
            # Penalize dof velocities
            dof_acc = -2.5e-7
            # Penalize dof accelerations
            action_rate = -0.01
            # Penalize changes in actions
            collision = -0.5
            # Penalize collisions on selected bodies
            termination = -0.0
            # Terminal reward / penalty
            dof_pos_limits = -0.01
            # Penalize dof positions too close to the limit
            dof_vel_limits = -0.01
            # Penalize dof velocities too close to the limit
            # clip to max error = 1 rad/s per joint to avoid huge penalties
            torque_limits = -0.01
            # penalize torques too close to the limit
            tracking_lin_vel = 20.0
            # Tracking of linear velocity commands (xy axes)
            x_afap = 1.0
            # linear velocity of x-axis
            tracking_x_vel = 1.0
            # Tracking of linear velocity in x-axis commands (xy axes)
            tracking_ang_vel = 0.
            # Tracking of angular velocity commands (yaw)
            feet_air_time = 1.0
            # Reward long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            stumble = -0.3
            # Penalize feet hitting vertical surfaces
            stand_still = -0.
            # Penalize motion at zero commands
            feet_contact_forces = -0.01
            # penalize high contact forces
            # forward = 20.
            # lateral_movement_and_rotations = -21.
            # work = -0.002
            # ground_impact = -0.02
            # smoothness = -0.001
            # action_magnitude = -0.07
            # joint_speed = -0.002
            # z_acc = -2.0
            # foot_slip = -0.8

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100.  # forces above this value are penalized

    class termination:
        collision = True
        yaw = True
        pitch = True
        roll = True

        # unit pi
        yaw_limit = 1 / 3
        pitch_limit = 1 / 3
        roll_limit = 1 / 3


class LeggedDynamicConfig(LeggedVisualInputConfig):
    class env(LeggedVisualInputConfig.env):
        episode_length_s = 40  # episode length in seconds
        unflatten_terrain = False
        env_spacing = 15

    class commands:
        # only move forward
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 20.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [.4, .8]  # min max [m/s]
            lin_vel_y = [-.0, .0]  # min max [m/s]
            ang_vel_yaw = [-.0, 0.0]  # min max [rad/s]
            heading = [-.0, 0.]

    class rewards:
        # copied from the legged robot config
        class scales:
            feet_stumble = 0

            lin_vel_z = 0
            # Penalize z axis base linear velocity
            ang_vel_xy = 0
            # Penalize xy axes base angular velocity
            orientation = 1.5
            # Penalize non flat base orientation
            base_height = 0
            # Penalize base height away from target
            torques = 0
            # Penalize torques
            dof_vel = 0
            # Penalize dof velocities
            dof_acc = 0
            # Penalize dof accelerations
            action_rate = 0
            # Penalize changes in actions
            collision = 3.0
            # Penalize collisions on selected bodies
            termination = 0
            # Terminal reward / penalty
            dof_pos_limits = 0
            # Penalize dof positions too close to the limit
            dof_vel_limits = 0
            # Penalize dof velocities too close to the limit
            # clip to max error = 1 rad/s per joint to avoid huge penalties
            torque_limits = 0
            # penalize torques too close to the limit
            tracking_lin_vel = 0
            # Tracking of linear velocity commands (xy axes)
            x_afap = 0
            # linear velocity of x-axis
            tracking_x_vel = 0
            # Tracking of linear velocity in x-axis commands (xy axes)
            tracking_ang_vel = 0
            # Tracking of angular velocity commands (yaw)
            feet_air_time = 0
            # Reward long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            stumble = 0
            # Penalize feet hitting vertical surfaces
            stand_still = 0
            # Penalize motion at zero commands
            feet_contact_forces = 0
            forward = 1.0
            energy = 0.04
            alive = 1.0
            # penalize high contact forces
            # forward = 20.
            # lateral_movement_and_rotations = -21.
            # work = -0.002
            # ground_impact = -0.02
            # smoothness = -0.001
            # action_magnitude = -0.07
            # joint_speed = -0.002
            # z_acc = -2.0
            # foot_slip = -0.8

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100.  # forces above this value are penalized

    class termination:
        collision = False
        yaw = False
        pitch = False
        roll = False

        # unit pi
        yaw_limit = 1 / 3
        pitch_limit = 1 / 3
        roll_limit = 1 / 3

    class pedestrians:
        num = 10
        max_speed = 0.5
        length = 0.3
        width = 0.3
        height = 1.5


class LeggedDigitalTwinEnvConfig(LeggedVisualInputConfig):
    class env(A1RoughCfg.env):
        asset_file = ""
        asset_root = ""
        asset_scale = 2.3
        env_spacing = 0.5
        extra_distance = 1000

    class terrain(A1RoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = True

    # class init_state(LeggedVisualInputConfig.init_state):
    #     pos = [1000, 0, 0.42] # x,y,z [m]


class LeggedMultiRobConfig(LeggedVisualInputConfig):
    class env(LeggedVisualInputConfig.env):
        episode_length_s = 40  # episode length in seconds
        unflatten_terrain = False
        env_spacing = 15

    class commands:
        # only move forward
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 20.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [.4, .8]  # min max [m/s]
            lin_vel_y = [-.1, .1]  # min max [m/s]
            ang_vel_yaw = [-.1, 0.1]  # min max [rad/s]
            heading = [-0.5, 0.5]

    class rewards:
        # copied from the legged robot config
        class scales:
            feet_stumble = -0.0

            lin_vel_z = -2.0
            # Penalize z axis base linear velocity
            ang_vel_xy = .0
            # Penalize xy axes base angular velocity
            orientation = -0.5
            # Penalize non flat base orientation
            base_height = -1
            # Penalize base height away from target
            torques = -0.0002
            # Penalize torques
            dof_vel = -0.
            # Penalize dof velocities
            dof_acc = -2.5e-7
            # Penalize dof accelerations
            action_rate = -0.01
            # Penalize changes in actions
            collision = -0.5
            # Penalize collisions on selected bodies
            termination = -0.0
            # Terminal reward / penalty
            dof_pos_limits = -0.01
            # Penalize dof positions too close to the limit
            dof_vel_limits = -0.01
            # Penalize dof velocities too close to the limit
            # clip to max error = 1 rad/s per joint to avoid huge penalties
            torque_limits = -0.01
            # penalize torques too close to the limit
            tracking_lin_vel = 20.0
            # Tracking of linear velocity commands (xy axes)
            x_afap = 1.0
            # linear velocity of x-axis
            tracking_x_vel = 1.0
            # Tracking of linear velocity in x-axis commands (xy axes)
            tracking_ang_vel = 0.
            # Tracking of angular velocity commands (yaw)
            feet_air_time = 1.0
            # Reward long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            stumble = -0.3
            # Penalize feet hitting vertical surfaces
            stand_still = -0.
            # Penalize motion at zero commands
            feet_contact_forces = -0.01
            # penalize high contact forces
            # forward = 20.
            # lateral_movement_and_rotations = -21.
            # work = -0.002
            # ground_impact = -0.02
            # smoothness = -0.001
            # action_magnitude = -0.07
            # joint_speed = -0.002
            # z_acc = -2.0
            # foot_slip = -0.8

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100.  # forces above this value are penalized

    class termination:
        collision = False
        yaw = False
        pitch = False
        roll = False

        # unit pi
        yaw_limit = 1 / 3
        pitch_limit = 1 / 3
        roll_limit = 1 / 3

    class pedestrians:
        num = 10
        max_speed = 0.5
        length = 0.3
        width = 0.3
        height = 1.5

    # class asset:
    #     self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
