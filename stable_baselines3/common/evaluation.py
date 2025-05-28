import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
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
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs, ), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations, state=states, episode_start=episode_starts, deterministic=deterministic
        )
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def evaluate_policy_grid_obs(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    return_AUC: bool = True,
    return_Accuracy: bool = True,
    warn: bool = True,
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
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    # if not isinstance(env, VecEnv):
    #     env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    # NOTE: initial step
    # observations, rewards, dones, infos = env.reset()  # [num_env, obs_size], reset() in dict to array (self._gym_env.reset())
    observations, rewards, dones, infos, accuracies = env.reset()  # [num_env, obs_size], reset() in dict to array (self._gym_env.reset())
    rewards = rewards.to('cpu')

    # NOTE: align with the args_eval.num_envs in xxx_eval.py
    n_envs = rewards.shape[0]
    assert n_eval_episodes % n_envs == 0, "n_envs must be divisible by n_eval_episodes"

    max_length = 100

    episode_rewards = []
    episode_lengths = []
    episode_accuracies = []

    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = torch.tensor([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=torch.int) # [n_envs], episode_count_targets.sum() == n_eval_episodes
    episode_counts = torch.zeros(n_envs, dtype=torch.uint8)
    episode_starts = torch.ones((n_envs, ), dtype=torch.bool)

    current_rewards = torch.zeros(n_envs)
    current_lengths = torch.zeros(n_envs, dtype=torch.uint8)

    current_rewards += rewards.to('cpu')    # accumulated reward


    if not return_AUC:  # debug
        print("BUG!!!!!")
        exit()
        while (episode_counts < episode_count_targets).any():
            actions, _ = model.predict(
                observations, state=None, deterministic=deterministic
            )
            observations, rewards, dones, infos = env.step(actions)     # step() in _worker()

            current_rewards += rewards  # rewards.shape: (n_envs)
            current_lengths += 1

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:

                    # NOTE: unpack values so that the callback can access the local variables. (e.g. locals_["info"] in callbacks.py)
                    reward = rewards[i]
                    done = dones[i]
                    episode_starts[i] = done
                    if n_envs == 1:
                        info = infos[i]
                    elif isinstance(infos, dict):   # n_envs > 1, eval_env == 1
                        info = infos
                    else:
                        info = infos[0]

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                        else:   # <-
                            episode_rewards.append(current_rewards[i].clone())
                            episode_lengths.append(current_lengths[i].clone())
                            episode_counts[i] += 1

                        # NOTE: initial step and corresponding info (rewards, ...) would be computed in next step
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            if render:
                env.render()
    else:

        AUC_rews = torch.zeros(n_eval_episodes, max_length)
        episode_done_flag = torch.zeros(n_eval_episodes)    # num_env * num_repeat
        episode_done_length = torch.zeros(n_eval_episodes)    # num_env * num_repeat

        AUC_rews = AUC_update(AUC_rews.clone(), rewards.clone(), current_lengths, episode_done_flag)    # initial step

        current_lengths += 1

        while (episode_counts < episode_count_targets).any():
            actions, _ = model.predict(observations, state=None, deterministic=deterministic)
            # observations, rewards, dones, infos = env.step(actions)     # step() in _worker()
            observations, rewards, dones, infos, accuracies = env.step(actions)     # step() in _worker(), accuracy

            AUC_rews = AUC_update(AUC_rews.clone(), rewards.clone(), current_lengths, episode_done_flag)    # NOTE: AUC computation

            current_rewards += rewards  # rewards.shape: (n_envs)
            current_lengths += 1

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:

                    # NOTE: unpack values so that the callback can access the local variables. (e.g. locals_["info"] in callbacks.py)
                    reward = rewards[i]
                    done = dones[i]
                    episode_starts[i] = done

                    if n_envs == 1:
                        info = infos[i]
                    elif isinstance(infos, dict):   # n_envs > 1, eval_env == 1
                        info = infos
                    else:
                        info = infos[0]

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[i]:
                        if is_monitor_wrapped:  # false
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                                episode_done_flag[episode_counts[i].item()*n_envs+i] += 1.
                                episode_done_length[episode_counts[i].item()*n_envs+i] = current_lengths[i].item()
                        else:   # <-
                            accuracy = accuracies[str(i)]
                            episode_rewards.append(current_rewards[i].clone())
                            episode_lengths.append(current_lengths[i].clone())
                            episode_accuracies.append(accuracy)   # the order doesn't matter
                            episode_done_flag[episode_counts[i].item()*n_envs+i] += 1.
                            episode_done_length[episode_counts[i].item()*n_envs+i] = current_lengths[i].item()
                            episode_counts[i] += 1

                        # NOTE: initial step and corresponding info (rewards, ...) would be computed in next step
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            if render:
                env.render()

    mean_AUC = sum([AUC_rews[:, idx] * (max_length - idx) for idx in range(max_length)]) / max_length   # [n_envs]
    # mean_AUC = torch.stack([AUC_rews[:, idx] * (max_length - idx) for idx in range(max_length)], dim=1).sum(dim=1) / max_length   # [n_envs]
    mean_reward = np.mean(episode_rewards)  # rewards at the end of episodes from eval_envs
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"


    if return_AUC & return_Accuracy:
        return episode_rewards, episode_lengths, mean_AUC, episode_accuracies
    if return_AUC:
        return episode_rewards, episode_lengths, mean_AUC
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

def AUC_update(AUC_rews, cur_rewards, cur_lengths, episode_done_flag):
    """
    Params:
        # AUC_rews: (n_envs, max_length)
        # cur_lengths: scaler
        # episode_done_flag: (n_envs), if done in this episode\
        # dones: (n_envs), if done after this step

        AUC_rews: (n_eval_episodes, 500)
        cur_rewards: (n_envs)
        cur_lengths: (n_envs), from 0 to max_length-1 within one episode
        episode_done_flag: (n_eval_episodes)
        n_rounds
    """
    n_envs = cur_rewards.shape[0]
    n_finished_episode = [episode_done_flag[env_idx::n_envs].sum() for env_idx in range(n_envs)]    # [n_envs]

    for env_episode_idx in range(episode_done_flag.shape[0]):
        env_idx = env_episode_idx % n_envs
        # if episode_done_flag[env_episode_idx]:
        #     AUC_rews[env_episode_idx, cur_lengths[env_idx].item()] = AUC_rews[env_episode_idx, cur_lengths[env_idx].item() - 1]
        # else:
        if (not episode_done_flag[env_episode_idx]) and (env_episode_idx == (n_finished_episode[env_idx] * n_envs + env_idx)):
            if cur_lengths[env_idx].item() >= AUC_rews.shape[0]:
                break
            AUC_rews[env_episode_idx, cur_lengths[env_idx].item()] = cur_rewards[env_idx].cpu().clone()

    return AUC_rews
