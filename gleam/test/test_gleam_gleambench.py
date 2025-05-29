import os
from isaacgym import gymapi
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args
from gleam.wrapper.env_wrapper_gleam import EnvWrapperGLEAM
from gleam.network.encoder import Encoder_GLEAM
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks_gleam import EvalCallback_GLEAM
from stable_baselines3.common.evaluation_gleam import evaluate_policy_grid_obs
from stable_baselines3.common.policies import ActorCriticPolicy_Discrete_Eval
from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
from stable_baselines3.utils import get_time_str
from wandb_utils import team_name, project_name
from wandb_utils.wandb_callback import WandbCallback

OPEN_ROBOT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def main():
    additional_args = [
        {
            "name": "--eval_device",
            "type": str,
            "default": "cuda:0",
        },
        {
            "name": "--buffer_size",
            "type": int,
            "default": 30,
            "help": "length of buffer"
        },
        {
            "name": "--n_steps",
            "type": int,
            "default": 512,       # 32 * 512 = 16384
            "help": "number of steps to collect in each env"
        },
        {
            "name": "--batch_size",
            "type": int,
            "default": 128,
            "help": "SGD batch size"
        },
        {
            "name": "--save_freq",
            "type": int,
            "default": 50000,
            "help": "save the model per <save_freq> iter"
        },
        {
            "name": "--total_iters",
            "type": int,
            "default": 1250,
            "help": "the number of training iters"
        },
        {
            "name": "--n_epochs",
            "type": int,
            "default": 5
        },
        {
            "name": "--use_target_kl",  # LQY: need search a good kl
            "type": bool,
            "default": True
        },
        {
            "name": "--target_kl",
            "type": float,
            "default": 0.05
        },
        {
            "name": "--vf_coeff",
            "type": float,
            "default": 0.8
        },
        {
            "name": "--ent_coeff",
            "type": float,
            "default": 0.01
        },
        {
            "name": "--lr",
            "type": float,
            "default": 0.0
        },
        {
            "name": "--unflatten_terrain",
            "type": bool,
            "default": False},
        {
            "name": "--first_view_camera",
            "type": bool,
            "default": False},
    ]
    reward_args = [
        {
            "name": "--surface_coverage",
            "type": float,
            "default": 1.0,
            "help": "surface coverage ratio"
        },
        {
            "name": "--only_positive_rewards",
            "type": bool,
            "default": False,
            "help": "If true negative total rewards are clipped at zero (avoids early termination problems)"
        },
        {
            "name": "--max_contact_force",
            "type": float,
            "default": 100,
            "help": "Forces above this value are penalized"
        },
    ]

    args = get_args(additional_args+reward_args)
    args.task = "eval_gleam_gleambench"

    use_wandb = False   # local inference
    # args.headless = True
    # args.num_envs = 32

    # # debug
    # # args.headless = False  # False: visualization
    # args.num_envs = 2
    # use_wandb = False

    ckpt_path = args.ckpt_path
    print(ckpt_path)
    assert ckpt_path is not None, "Please specify the checkpoint path!"
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist!"


    exp_name = args.task
    seed = 0
    trial_name = f"{exp_name}_{get_time_str()}" \
        if args.exp_name is None or len(args.exp_name) == 0 \
        else f"{args.exp_name}_{get_time_str()}"
    log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs", trial_name)
    print("[LOGGING] We start logging training data into {}".format(log_dir))

    # ===== Setup the evaluation environment =====
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.visual_input.stack = args.buffer_size

    # evaluation env
    env_eval, env_cfg_eval = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_cfg_dict = {key:value for key, value in env_cfg_eval.__dict__.items()}

    env = EnvWrapperGLEAM(env_eval)

    # ===== Setup the config =====
    config = dict(
        algo=dict(
            policy=ActorCriticPolicy_Discrete_Eval,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=Encoder_GLEAM,
                features_extractor_kwargs=dict(
                    encoder_param={
                        "hidden_shapes": [256, 256],
                        "visual_dim": 256
                    },
                    net_param={
                        "transformer_params": [[1, 256], [1, 256]],
                        "append_hidden_shapes": [256, 256]
                    },
                    state_input_shape=(args.buffer_size * 6,),  # buffer_size * pose_size
                    visual_input_shape=(
                        1, 128, 128
                    )
                )
            ),
            env=env,
            learning_rate=args.lr,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=args.target_kl if args.use_target_kl else None,
            max_grad_norm=1,
            n_steps=args.n_steps,  # steps to collect in each env
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=0.2,
            vf_coef=args.vf_coeff,
            clip_range_vf=0.2,
            ent_coef=args.ent_coeff,
            tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device=args.sim_device,
        ),

        # Meta data
        gpu_simulation=True,
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the callbacks =====
    callbacks = [
        EvalCallback_GLEAM(
            eval_env=env_eval,
            n_eval_episodes=10000,
            log_path=log_dir,
            eval_freq=1000000000,
            deterministic=False,
            render=True,
            verbose=1,
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(trial_name=trial_name, exp_name=exp_name, project_name=project_name, config={**config, **env_cfg_dict})
        )
    callbacks = CallbackList(callbacks)

    # ===== Launch evaluation =====
    model = PPO_Grid_Obs(**config["algo"])
    if ckpt_path:
        model.set_parameters(ckpt_path)

    evaluate_policy_grid_obs(
        model=model,
        env=env_eval,
        deterministic=True,
    )
    print(ckpt_path)
    print("Done.")


if __name__ == '__main__':
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()

    main()

    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Total wall-clock time: {:.3f}min".format((t_end-t_start)/60))
