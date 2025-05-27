import logging
import os

import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
from stable_baselines3.utils import get_api_key_file

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class WandbCallback(BaseCallback):
    """ Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used
    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """
    def __init__(
        self,
        trial_name,
        exp_name,
        project_name,
        config=None,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
    ):
        team_name = config["team_name"]
        # PZH: Setup our key
        WANDB_ENV_VAR = "WANDB_API_KEY"
        key_file_path = get_api_key_file(
            "wandb_lqy_key_file.text" if team_name == "lqy0057" else None
        )  # Search ~/wandb_api_key_file.txt first, then use PZH's
        with open(key_file_path, "r") as f:
            key = f.readline()
        key = key.replace("\n", "")
        key = key.replace(" ", "")
        os.environ[WANDB_ENV_VAR] = key

        self.run = wandb.init(
            id=trial_name,
            # id=exp_name,
            # name=run_name,
            config=config or {},
            resume=True,
            reinit=True,
            # allow_val_change=True,
            group=exp_name,
            project=project_name,
            entity=team_name,
            sync_tensorboard=True,  # Open this and setup tb in sb3 so that we can get log!
            save_code=False
        )

        super(WandbCallback, self).__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.gradient_save_freq = gradient_save_freq

        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info("Saving model checkpoint to " + self.path)
