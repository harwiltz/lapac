import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lapac.actor_network import ActorNetwork
from lapac.critic_network import CriticNetwork
from lapac.model_network import ModelNetwork
from lapac.trajectory import StepType

class LapacAgent(object):
    def __init__(
            self,
            observation_space,
            action_space,
            sequence_length=4,
            feature_size=256,
            gamma=0.99,
            tau=0.005,
            actor_base_depth=256,
            critic_base_depth=256,
            z1_size=32,
            z2_size=256,
            initial_log_ent=0.0,
            model_lr=1e-4,
            actor_lr=3e-4,
            critic_lr=3e-4,
            ent_lr=3e-4,
            action_stickyness=2):

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._training_iterations = 0
        self._decisions_made = 0
        self._last_action = None

        self._observation_space = observation_space
        self._action_space = action_space
        self._feature_size = feature_size
        self._gamma = gamma
        self._tau = tau
        self._action_stickyness=action_stickyness
        self._sequence_length = sequence_length

        self._model_network = ModelNetwork(
                observation_space,
                action_space,
                feature_size=feature_size,
                z1_size=z1_size,
                z2_size=z2_size).to(self._device)
        self._actor_network = ActorNetwork(
                feature_size,
                action_space,
                sequence_length,
                base_depth=actor_base_depth).to(self._device)
        self._critic_network = CriticNetwork(
                z1_size + z2_size,
                action_space,
                base_depth=critic_base_depth).to(self._device)
        self._target_critic_network = copy.deepcopy(self._critic_network).to(self._device)

        # Weight of entropy for MaxEnt RL
        # Train it like SAC and SLAC
        self._log_ent = nn.Parameter(torch.FloatTensor([initial_log_ent]))
        self._target_ent = torch.Tensor([-self._action_space.shape[0]])

        self._model_optimizer = torch.optim.Adam(self._model_network.parameters(), lr=model_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic_network.parameters(), lr=critic_lr)
        self._actor_optimizer = torch.optim.Adam(self._actor_network.parameters(), lr=actor_lr)
        self._ent_optimizer = torch.optim.Adam([self._log_ent], lr=ent_lr)

    def action(self, context):
        images, actions, step_types = context
        if self._decisions_made % self._action_stickyness == 0:
            images = torch.FloatTensor(images).to(self._device)
            actions = torch.FloatTensor(actions).to(self._device)
            features = self._model_network.feature_extractor(images)
            next_action_dist = self._actor_network(features, actions)
            next_action = next_action_dist.sample()
            self._last_action = next_action
        else:
            next_action = self._last_action
        self._decisions_made += 1
        return next_action

    def train(self, experience):
        self._training_iterations += 1

        model_loss, model_artifacts = self._model_network.compute_loss(experience)

        z1_samples = model_artifacts['z1_posterior_samples']
        z2_samples = model_artifacts['z2_posterior_samples']
        latent_posterior_samples = torch.cat([z1_samples, z2_samples], axis=-1)

        images, actions, rewards, step_types = experience
        features = model_artifacts['features']

        actor_loss, actor_artifacts = self.compute_actor_loss(
                features,
                actions,
                latent_posterior_samples)
        critic_loss, critic_artifacts = self.compute_critic_loss(
                features,
                actions,
                latent_posterior_samples,
                rewards)

        ent_loss = self.compute_ent_loss(features, actions)

        self._model_optimizer.zero_grad()
        model_loss.backward(retain_graph=True)
        self._model_optimizer.step()

        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self._critic_optimizer.step()

        self._ent_optimizer.zero_grad()
        ent_loss.backward()
        self._ent_optimizer.step()

        critic_params = self._critic_network.parameters()
        target_critic_params = self._target_critic_network.parameters()
        for params, target_params in zip(critic_params, target_critic_params):
            target_params.data.copy_(self._tau * params.data + (1-self._tau) * target_params.data)

        return {
            'model_loss': model_loss,
            'actor_loss': actor_loss,
            'actor_log_pi': actor_artifacts['log_pi'],
            'critic_loss': critic_loss,
            'ent_loss': ent_loss,
            'ent': torch.exp(self._log_ent),
            'images': model_artifacts['images'],
            'posterior_images': model_artifacts['posterior_images'],
            'reward_loss': model_artifacts['reward_loss'],
        }

    def plan(self, num_sequences=1):
        feature_batch = []
        latent_batch = []
        rew_batch = []
        action_batch = []
        for _ in range(num_sequences):
            z1_samples = []
            z2_samples = []
            features = [] # Needed only for policy sampling/learning
            img_ctx, action_ctx, rew_ctx, step_types = self.clear_context()
            z1 = self._model_network._z1_first_prior(action_ctx).sample()
            z2 = self._model_network._z2_first_prior(z1).sample()
            obs = self._model_network.decoder(torch.cat([z1, z2])).sample()
            for t in range(self._sequence_length):
                z1_samples.append(z1)
                z2_samples.append(z2)
                img_ctx = np.concatenate([img_ctx[1:], np.expand_dims(obs.numpy(), 0)])
                action = self.action((img_ctx, action_ctx, step_types))
                step_types = step_types[1:] + [StepType.mid]
                action_ctx = np.concatenate([action_ctx[1:], np.expand_dims(action.numpy(), 0)])
                feats = self._model_network.feature_extractor(obs)
                features.append(feats)
                z1_input = torch.cat([
                    z2,
                    action
                ])
                next_z1 = self._model_network._z1_prior(z1_input).sample()
                z2_input = torch.cat([
                    next_z1,
                    z2,
                    action
                ])
                next_z2 = self._model_network._z2_prior(z2_input).sample()
                rew_input = torch.cat([
                    torch.cat([z1, z2]),
                    action,
                    torch.cat([next_z1, next_z2])
                ])
                rew = self._model_network._reward_model(rew_input).sample().numpy()
                rew_ctx = np.append(rew_ctx[1:], rew)
                z1 = next_z1
                z2 = next_z2
                obs = self._model_network.decoder(torch.cat([z1, z2])).sample()
            z1_samples.append(z1)
            z2_samples.append(z2)
            features.append(self._model_network.feature_extractor(obs))
            z1_samples = torch.stack(z1_samples)
            z2_samples = torch.stack(z2_samples)
            latent_samples = torch.cat([z1_samples, z2_samples], axis=1)
            features = torch.stack(features)
            feature_batch.append(features)
            latent_batch.append(latent_samples)
            action_batch.append(action_ctx)
            rew_batch.append(rew_ctx)

        feature_batch = torch.stack(feature_batch)
        latent_batch = torch.stack(latent_batch)
        rew_batch = torch.FloatTensor(rew_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)

        actor_loss, _ = self.compute_actor_loss(feature_batch, action_batch, latent_batch)
        critic_loss, _ = \
            self.compute_critic_loss(feature_batch, action_batch, latent_batch, rew_batch)

        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self._critic_optimizer.step()

        return {
            'planning_actor_loss': actor_loss,
            'planning_critic_loss': critic_loss,
        }

    def compute_actor_loss(self, features, actions, latent_posterior_samples):
        # actions shape: (Batch, Time, Action)
        policy_dist = self._actor_network(features, actions)
        next_actions = policy_dist.sample()
        q_values = torch.min(*self._critic_network(latent_posterior_samples[:, -1], next_actions))
        log_pis = policy_dist.log_prob(next_actions)
        loss = torch.exp(self._log_ent) * log_pis - q_values.squeeze()
        return loss.mean(), {'log_pi': log_pis.mean()}

    def compute_critic_loss(self, features, actions, latent_posterior_samples, rewards):
        # Q(z_t, a_t)
        estimated_q_values_1, estimated_q_values_2 = self._critic_network(
                latent_posterior_samples[:, -2],
                actions[:, -2])
        ent = torch.exp(self._log_ent)
        with torch.no_grad():
            next_actions_dist = self._actor_network(features, actions)
            next_actions = next_actions_dist.sample()
            log_pis = next_actions_dist.log_prob(next_actions)
            target_q_values = torch.min(
                *self._target_critic_network(latent_posterior_samples[:, -2], next_actions)
            ).squeeze()
            q_next = self._gamma * (target_q_values - ent * log_pis)
            target_q_values = rewards[:, -1] + q_next
        q_loss_1 = F.mse_loss(estimated_q_values_1.squeeze(), target_q_values)
        q_loss_2 = F.mse_loss(estimated_q_values_2.squeeze(), target_q_values)
        return q_loss_1 + q_loss_2, {}

    def compute_ent_loss(self, features, actions):
        with torch.no_grad():
            next_actions_dist = self._actor_network(features, actions)
            next_actions = next_actions_dist.sample()
            log_pis = next_actions_dist.log_prob(next_actions)
        return (self._log_ent * (-log_pis - self._target_ent)).mean()

    def clear_context(self):
        img_ctx = np.zeros([self._sequence_length + 1] + list(self._observation_space.shape))
        action_ctx = np.zeros([self._sequence_length] + list(self._action_space.shape))
        reward_ctx = np.zeros(self._sequence_length)
        step_types = [StepType.first for _ in range(self._sequence_length)]
        return img_ctx, action_ctx, reward_ctx, step_types

    @property
    def actor_network_input_shape(self):
        obs_size = self._feature_size
        act_size = self._action_space.shape[0]
        size = self._sequence_length * (obs_size + act_size) + obs_size
        return (size,)
