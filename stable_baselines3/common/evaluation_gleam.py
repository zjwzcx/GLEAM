import gym
import torch
from typing import List, Tuple, Union
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy_grid_obs(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    deterministic: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    with torch.no_grad():
        observations = env.reset()
        # set termination condition in eval_env
        while(1):
            actions, _ = model.predict(observations, state=None, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)

