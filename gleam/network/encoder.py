# from typing import Any, Dict, List, Optional, Type, Union, Tuple
from typing import List, Tuple

import gym
import torch as th
from gleam.network.base import LocoTransformerEncoder_Map
from gleam.network.locotransformer import LocoTransformer_GLEAM
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Encoder_GLEAM(BaseFeaturesExtractor):
    """
    We adapt the LocoTransformer to encode raw observations, including:
    - state input: historical poses within this episode.
    - visual input: current egocentric map, with the shape of (128, 128).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        encoder_param=None,
        net_param=None,
        visual_input_shape=None,
        state_input_shape=None,
    ):
        assert encoder_param is not None, "Need parameters !"
        assert net_param is not None, "Need parameters !"
        assert isinstance(visual_input_shape, List) or isinstance(visual_input_shape, Tuple), "Use tuple or list"
        assert isinstance(state_input_shape, List) or isinstance(state_input_shape, Tuple), "Use tuple or list"
        self.map_channel = visual_input_shape[0]
        self.map_shape = visual_input_shape[1:]
        self.state_input_shape = state_input_shape
        feature_dim = net_param["append_hidden_shapes"][-1]
        net_param["append_hidden_shapes"].pop()
        super(Encoder_GLEAM, self).__init__(observation_space, feature_dim)

        # create encoder, share encoder
        self.encoder = LocoTransformerEncoder_Map(
            in_channels=self.map_channel,
            state_input_dim=self.state_input_shape[0],
            **encoder_param
        )

        self.locotransformer = LocoTransformer_GLEAM(
            encoder=self.encoder,
            state_input_shape=self.state_input_shape[0],
            visual_input_shape=(self.map_channel, *self.map_shape),
            output_shape=feature_dim,
            **net_param
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        observations: [num_env, concat(state_input, visual_input)]
        """
        return self.locotransformer.forward(observations)