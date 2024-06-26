"""
In this script, we will implement the feature extractor for the actor and critic networks.
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
import torch.nn as nn
import torch as th

class ActorFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space,
                 features_dim: int = 0
                ) -> None:
        super(ActorFeatureExtractor, self).__init__(
            observation_space, 
            features_dim
            )
        self.flatten = nn.Flatten()    

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)
    

class CriticFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space,
                 features_dim: int = 0
                ) -> None:
        super(CriticFeatureExtractor, self).__init__(
            observation_space, 
            features_dim
            )
        self.flatten = nn.Flatten()    

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)