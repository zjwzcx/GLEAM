# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An env wrapper that flattens the observation dictionary to an array."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gym
import collections

import torch
from gym import spaces
import numpy as np

def flatten_observations(observation_dict, key_sequence):
    """Flattens the observation dictionary to an array.

    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_dict[key]) in observation_excluded,
    and ('other': the flattened array).

    Args:
      observation_dict: A dictionary of all the observations.
      key_sequence: A list/tuple of all the keys of the observations to be
        added during flattening.

    Returns:
      An array or a dictionary of observations based on whether
        observation_excluded is empty.
    """
    observations = []
    for key in key_sequence:
        value = observation_dict[key]
        observations.append(value)

    flat_observations = torch.concat(observations, dim=-1)  # grid observation
    return flat_observations


def flatten_observation_spaces(observation_spaces, key_sequence):
    """Flattens the dictionary observation spaces to gym.spaces.Box.

    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_spaces[key]) in observation_excluded,
    and ('other': the flattened Box space).

    Args:
      observation_spaces: A dictionary of all the observation spaces.
      key_sequence: A list/tuple of all the keys of the observations to be
        added during flattening.

    Returns:
      A box space or a dictionary of observation spaces based on whether
        observation_excluded is empty.
    """
    assert isinstance(key_sequence, list)
    lower_bound = []
    upper_bound = []
    for key in key_sequence:
        value = observation_spaces.spaces[key]
        if isinstance(value, spaces.Box):
            lower_bound.append(np.asarray(value.low).flatten())
            upper_bound.append(np.asarray(value.high).flatten())
        else:
            continue

    lower_bound = np.concatenate(lower_bound)
    upper_bound = np.concatenate(upper_bound)
    observation_space = spaces.Box(np.array(lower_bound, dtype=np.float32), np.array(upper_bound, dtype=np.float32), dtype=np.float32)
    return observation_space


class EnvWrapperGLEAM(gym.Env):
    """An env wrapper that flattens the observation dictionary to an array."""
    def __init__(self, gym_env, observation_excluded=()):
        """Initializes the wrapper."""
        self.observation_excluded = observation_excluded
        self._gym_env = gym_env
        self.observation_space = self._flatten_observation_spaces(self._gym_env.observation_space)
        self.action_space = self._gym_env.action_space

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def _flatten_observation_spaces(self, observation_spaces):
        flat_observation_space = flatten_observation_spaces(
            observation_spaces=observation_spaces, key_sequence=["state", "ego_map_2D"]
        )
        return flat_observation_space

    def _flatten_observation(self, input_observation):
        """Flatten the dictionary to an array."""
        return flatten_observations(observation_dict=input_observation, key_sequence=["state", "ego_map_2D"])

    def reset(self):
        observation = self._gym_env.reset()
        return self._flatten_observation(observation)

    def step(self, action):
        """Steps the wrapped environment.

        Args:
          action: Numpy array. The input action from an NN agent.

        Returns:
          The tuple containing the flattened observation, the reward, the epsiode
            end indicator.
        """
        observation_dict, reward, done, _ = self._gym_env.step(action)
        return self._flatten_observation(observation_dict), reward, done, _

    def render(self, mode='human'):
        return self._gym_env.render(mode)

    def close(self):
        self._gym_env.close()
