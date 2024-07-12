from torch import nn
from typing import Type, Union, List, Dict, Tuple
import torch as th

from stable_baselines3.common.utils import get_device

class NonSharedMLPExtractor(nn.Module):
    """
    A feed forward neural network (multi-layer perceptron) with non shared parameters.

    :param observation_space: Observation space
    :param net_arch: Specification of the network architecture. This can be a list of integers or a list of dicts
        specifying the arguments for the layers (see input of `create_mlp`)
    :param activation_fn: Activation function
    """

    def __init__(
            self, 
            features_dim: dict, # {"pi": int, "vf": int}
            net_arch: Union[List[int], List[Dict[str, int]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = 'auto',
        ) -> None:
        super(NonSharedMLPExtractor, self).__init__()

        self.features_dim = features_dim
        print("features_dim: ", features_dim)

        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = features_dim["pi"]
        last_layer_dim_vf = features_dim["vf"]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)


    def forward(self, pi_features: th.Tensor, vf_features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(pi_features), self.forward_critic(vf_features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
