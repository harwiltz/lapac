import torch
import torch.nn as nn
import torch.distributions as dtb

from pgm_slac.utils import MultivariateNormalDiag

class ActorNetwork(nn.Module):
    def __init__(
            self,
            observation_size,
            action_space,
            sequence_length,
            base_depth=256):
        super(ActorNetwork, self).__init__()
        action_high = torch.Tensor(action_space.high)
        action_low = torch.Tensor(action_space.low)
        self._base_dist = MultivariateNormalDiag(
                sequence_length * (action_space.shape[0] + observation_size) + observation_size,
                base_depth,
                action_space.shape[0])
        """
        Distribution transformations to force sampled actions to be within the action space
        boundaries. We first use a sigmoid transform to map each dimension to (0,1), and then
        use an affine transformation to fit the desired range:
            SIGMOID[i] = Sigmoid(SAMPLED ACTION)[i] (in (0,1))
            ACTION[i] = LOW[i] + (HIGH[i] - LOW[i]) * SIGMOID[i] (in (LOW, HIGH))
        """
        self._transforms = [
                dtb.transforms.SigmoidTransform(),
                dtb.transforms.AffineTransform(
                    loc=action_low,
                    scale=(action_high - action_low))]

    def forward(self, features, actions):
        features_flat = features.reshape(list(features.shape[:-2]) + [-1])
        actions_flat = actions.reshape(list(actions.shape[:-2]) + [-1])
        actor_input = torch.cat([features_flat, actions_flat], axis=-1)
        base_dist = self._base_dist(actor_input)
        return dtb.TransformedDistribution(base_dist, self._transforms)
