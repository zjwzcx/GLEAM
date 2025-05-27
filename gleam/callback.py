import os
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback


class BestCKPTCallback(CheckpointCallback):
    def __init__(
        self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, key_list: list = None
    ):
        key_list = key_list or []
        self.rollout_count = 0
        super(BestCKPTCallback, self).__init__(save_freq, save_path, name_prefix, verbose)
        self.key_highest_value = {k: 0. for k in key_list}

    @property
    def best_save_path(self):
        return self.locals["self"].logger.dir

    @property
    def device(self):
        return self.locals["self"].device

    def _on_rollout_end(self) -> None:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        for key in self.key_highest_value.keys():
            value = self.calculate_value(key)
            if value > self.key_highest_value[key]:
                self.key_highest_value[key] = value
                path = os.path.join(self.best_save_path, "{}_best_{}".format(self.name_prefix, key))
                self.model.save(path)
                # if self.verbose > 1:
                print("Saving Best {} checkpoint to {}: {}".format(key, path, value))

    def calculate_value(self, key):
        ep_info_buffer = self.locals["self"].ep_info_buffer
        infotensor = th.tensor([], device=self.device)
        assert key in ep_info_buffer[0], "no key named {}, can not save checkpoint".format(key)
        for ep_info in ep_info_buffer:
            # handle scalar and zero dimensional tensor infos
            if not isinstance(ep_info[key], th.Tensor):
                ep_info[key] = th.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = th.cat((infotensor, ep_info[key].to(self.device)))
        value = th.mean(infotensor).detach().cpu().numpy()
        return value

class ReconstructionCallBack(BestCKPTCallback):
    pass
