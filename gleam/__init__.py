from legged_gym.utils.task_registry import task_registry

from gleam.env.env_gleam_stage1 import Env_GLEAM_Stage1
from gleam.env.env_gleam_stage2 import Env_GLEAM_Stage2
from gleam.env.env_gleam_eval import Env_GLEAM_Eval
from gleam.env.config_gleam import Config_GLEAM, DroneCfgPPO
from gleam.env.config_gleam_eval import Config_GLEAM_Eval
task_registry.register("train_gleam_stage1", Env_GLEAM_Stage1, Config_GLEAM, DroneCfgPPO)
task_registry.register("train_gleam_stage2", Env_GLEAM_Stage2, Config_GLEAM, DroneCfgPPO)
task_registry.register("eval_gleam_gleambench", Env_GLEAM_Eval, Config_GLEAM_Eval, DroneCfgPPO)
