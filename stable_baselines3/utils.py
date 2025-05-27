import os
import datetime
from legged_gym.env.base.base_task import BaseTask
from legged_gym import OPEN_ROBOT_ROOT_DIR
from gleam.wrapper.env_wrapper_gleam import EnvWrapperGLEAM


root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_api_key_file(wandb_key_file=None):
    wandb_key_file = wandb_key_file or "wandb_api_key_file.txt"
    root = OPEN_ROBOT_ROOT_DIR
    path = os.path.join(root, "wandb_utils", wandb_key_file)
    print("We are using this wandb key file: ", path)
    return path


def get_time_str():
    return datetime.datetime.now().strftime("%m%d_%H%M_%S")


def is_isaac_gym_env(env):
    Isacc_Gym_Env = [
        BaseTask, 
        EnvWrapperGLEAM,
    ]

    for target_env in Isacc_Gym_Env:
        if isinstance(env, target_env):
            return True
    return False
