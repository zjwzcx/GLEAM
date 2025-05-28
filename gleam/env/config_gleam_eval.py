from gleam.env.config_legged_visual import LeggedVisualInputConfig
from isaacgym import gymapi


class Config_GLEAM_Eval(LeggedVisualInputConfig):
    position_use_polar_coordinates = False  # the position will be represented by (r, \theta, \phi) instead of (x, y, z)
    direction_use_vector = False  # (r, p, y)
    debug_save_image_tensor = False
    debug_save_path = None
    max_episode_length = 50    # max_steps_per_episode

    num_sampled_point = 5000

    class rewards:
        class scales:   # * self.dt (0.019999) / episode_length_s (20). For example, when reward_scales=1000, rew_xxx = reward * 1000 * 0.019999 / 20 = reward
            surface_coverage_2d = 1000     # original scale (coverage ratio: [0, 1])

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 100.    # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        pi = 3.14159265359
        clip_observations = 255.

        # discrete action space
        init_action = [64, 64, 0, 0, 0, 0]  # 64: keep still
        clip_actions_up = [128, 128, 0, 0, 0, 0]
        clip_actions_low = [0, 0, 0, 0, 0, 0]

    class visual_input(LeggedVisualInputConfig.visual_input):
        camera_width = 256
        camera_height = 32
        horizontal_fov = 90.0   # Horizontal field of view in degrees. Vertical field of view is calculated from height to width ratio

        stack = 100
        supersampling_horizontal = 1
        supersampling_vertical = 1
        normalization = True    # normalize pixels to [0, 1]
        far_plane = 2000000.0   # distance in world coordinates to far-clipping plane
        near_plane = 0.0010000000474974513  # distance in world coordinate units to near-clipping plane

        type = gymapi.IMAGE_DEPTH

    class asset(LeggedVisualInputConfig.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/drone/cf2x.urdf'
        name = "cf2x"
        prop_name = "prop"
        penalize_contacts_on = ["prop", "base"]
        terminate_after_contacts_on = ["prop", "base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class env(LeggedVisualInputConfig.env):
        num_observations = 6
        episode_length_s = 20  # in second !!!!!
        num_actions = 6
        env_spacing = 20

    class termination:
        collision = True
        max_step_done = True