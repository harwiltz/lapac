import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(
            self,
            latent_size,
            action_space,
            base_depth=256):
        super(CriticNetwork, self).__init__()
        action_size = action_space.shape[0]
        self._l_1_1 = nn.Linear(latent_size + action_size, base_depth)
        self._l_1_2 = nn.Linear(base_depth, base_depth)
        self._l_1_3 = nn.Linear(base_depth, 1)
        self._l_2_1 = nn.Linear(latent_size + action_size, base_depth)
        self._l_2_2 = nn.Linear(base_depth, base_depth)
        self._l_2_3 = nn.Linear(base_depth, 1)

    def call(self, features, action):
        critic_input = torch.cat([features, action], axis=-1)

        q1 = self._l_1_1(critic_input)
        q1 = F.relu(q1)
        q1 = self._l_1_2(q1)
        q1 = F.relu(q1)
        q1 = self._l_1_3(q1)

        q2 = self._l_2_1(critic_input)
        q2 = F.relu(q2)
        q2 = self._l_2_2(q2)
        q2 = F.relu(q2)
        q2 = self._l_2_3(q2)

        return q1, q2
