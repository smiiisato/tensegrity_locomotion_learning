"""
This script contains the implementation of the Actor-Critic policy with independent network designs for the Actor and Critic.
"""

from stable_baselines3.common.policies import ActorCriticPolicy
import collections
import copy
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from feature_extractor import ActorFeatureExtractor, CriticFeatureExtractor
from mlp_extractor import NonSharedMLPExtractor

class NonSharedActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class (with both actor and critic) for Independent Actor-Critic networks.

    :param observation_space: Observation space -> Dict{"actor": Box, "critic": Box}
    :param action_space: Action space -> Box
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Network architecture
    :param activation_fn: Activation function
    :param ortho_init: Whether to use orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use a separate standard deviation for each action dimension
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = False, # In the original implementation, this is set to True
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            *args,
            **kwargs,
        ):
        super(NonSharedActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            *args,
            **kwargs,
        )

        # non-shared feature extractors
        self.pi_features_extractor = ActorFeatureExtractor(observation_space=observation_space.__getitem__("actor"),
                                                               features_dim=observation_space.__getitem__("actor").shape[0])
        self.vf_features_extractor = CriticFeatureExtractor(observation_space=observation_space.__getitem__("critic"),
                                                                features_dim=observation_space.__getitem__("critic").shape[0])
        
        # dimension of the features extracted by the feature extractor
        self.features_dim = {"pi": self.pi_features_extractor.features_dim, 
                             "vf": self.vf_features_extractor.features_dim}
        
        delattr(self, "features_extractor")  # remove the shared features extractor

        # rebuild the network with non-shared feature extractors
        self._rebuild(lr_schedule=lr_schedule)


    def _rebuild(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Rebuild the network (mainly for setting new optimizers with different learning rates)
        """
        self._build_mlp_extractor()
        self._build_actor_net()
        self._build_critic_net()

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.pi_features_extractor: np.sqrt(2),
                self.vf_features_extractor: np.sqrt(2),
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  
        

    def _build_mlp_extractor(self) -> None:
        """
        Build the separated actor and critic networks.
        """
        self.mlp_extractor = NonSharedMLPExtractor(
            features_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

    def _build_actor_net(self) -> None:
        """
        Build the actor network.
        """
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        # Action distribution
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")


    def _build_critic_net(self) -> None:
        """
        Build the critic network.
        """
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Linear(latent_dim_vf, 1)


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Compute the actor and critic output
        pi_features, vf_features = self.extract_features(obs)

        # get the latent feature of policy_obs and value_obs
        latent_pi, latent_vf = self.mlp_extractor(pi_features, vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob


    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Extract features in both actor and critic networks.

        :param obs: Observation
        :return: actor features, critic features
        """
        actor_obs, critic_obs = th.split(obs, [self.pi_features_extractor.features_dim, self.vf_features_extractor.features_dim], dim=1)
        return self.pi_features_extractor(actor_obs), self.vf_features_extractor(critic_obs)
    


    