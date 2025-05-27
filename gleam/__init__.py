from legged_gym.utils.task_registry import task_registry

from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1
from gleam.env.env_gleam_stage2 import Env_GLEAM_Stage2
from gleam.env.config_gleam import Config_GLEAM, DroneCfgPPO
task_registry.register("train_gleam_stage1", Env_GLEAM_Stage1, Config_GLEAM, DroneCfgPPO)
task_registry.register("train_gleam_stage2", Env_GLEAM_Stage2, Config_GLEAM, DroneCfgPPO)
