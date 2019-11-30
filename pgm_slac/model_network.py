import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dtb

from pgm_slac.utils import MultivariateNormalDiag
from pgm_slac.utils import FixedIsotropicNormal

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            base_depth=32,
            feature_size=256):
        super(FeatureExtractor, self).__init__()
        self._base_depth = base_depth
        self._conv1 = nn.Conv2d(3, self._base_depth, 5, stride=2)
        self._conv2 = nn.Conv2d(self._base_depth, 2 * self._base_depth, 3, stride=2)
        self._conv3 = nn.Conv2d(2 * self._base_depth, 4 * self._base_depth, 3, stride=2)
        self._conv4 = nn.Conv2d(4 * self._base_depth, 8 * self._base_depth, 3, stride=2)
        self._conv5 = nn.Conv2d(8 * self._base_depth, 8 * self._base_depth, 4)

    def forward(self, images):
        # (..., H, W, Channel) --> (Batch, H, W, Channel)
        data = torch.reshape(images, (-1, *images.shape[-3:]))
        # (Batch, H, W, Channel) --> (Batch, Channel, H, W)
        data = data.permute(0, 3, 1, 2)
        out = self._conv1(data)
        out = F.relu(out)
        out = self._conv2(out)
        out = F.relu(out)
        out = self._conv3(out)
        out = F.relu(out)
        out = self._conv4(out)
        out = F.relu(out)
        out = self._conv5(out)
        out = F.relu(out)
        return out.reshape(images.shape[:-3] + out.shape[-3:]).squeeze()

class Decoder(nn.Module):
    def __init__(
            self,
            latent_size,
            scale=0.1,
            base_depth=32,
            feature_size=256):
        super(Decoder, self).__init__()
        self._base_depth = base_depth
        self._scale = scale
        self._feature_size = feature_size
        self._formatter = nn.Linear(latent_size, feature_size)
        self._conv1 = nn.ConvTranspose2d(feature_size, 8 * self._base_depth, 4)
        self._conv2 = nn.ConvTranspose2d(8 * self._base_depth, 4 * self._base_depth, 3, stride=2)
        self._conv3 = nn.ConvTranspose2d(4 * self._base_depth, 2 * self._base_depth, 3, stride=2)
        self._conv4 = nn.ConvTranspose2d(2 * self._base_depth, self._base_depth, 3, stride=2)
        self._conv5 = nn.ConvTranspose2d(self._base_depth, 3, 5, stride=2)

    def forward(self, latents):
        # (..., Features, 1, 1) --> (Batch, Features, 1, 1)
        data = torch.reshape(latents, (-1, latents.shape[-1]))
        out = self._formatter(data).reshape(-1, self._feature_size, 1, 1)

        out = self._conv1(out)
        out = F.relu(out)
        out = self._conv2(out)
        out = F.relu(out)
        out = self._conv3(out)
        out = F.relu(out)
        out = self._conv4(out)
        out = F.relu(out)
        out = self._conv5(out)
        out = F.relu(out)
        # (Batch, Channel, H, W) --> (Batch, H, W, Channel)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(latents.shape[:-1] + out.shape[-3:])
        out = out.sigmoid()
        return dtb.Independent(dtb.normal.Normal(loc=out, scale=self._scale), 3)

class ModelNetwork(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            feature_size=256,
            z1_size=32,
            z2_size=256,
            reward_stddev=None):
        super(ModelNetwork, self).__init__()

        self.feature_extractor = FeatureExtractor(feature_size=feature_size)
        self.decoder = Decoder(z1_size + z2_size, feature_size=feature_size)

        self._feature_size = feature_size

        latent_mlp_hidden_size = 256
        self._z1_size = z1_size
        self._z2_size = z2_size
        self._full_latent_size = self._z1_size + self._z2_size

        self._action_size = action_space.shape[0]

        # z^1_1 ~ p(z^1_1) = N(0, I)
        self._z1_first_prior = FixedIsotropicNormal(z1_size)
        # z^2_1 ~ p(z^2_1 | z^1_1)
        self._z2_first_prior = MultivariateNormalDiag(
                self._z1_size,
                latent_mlp_hidden_size,
                self._z2_size)
        # z^1_{t+1} ~ p(z^1_{t+1} | z^2_t, a_t)
        self._z1_prior = MultivariateNormalDiag(
                self._z2_size + self._action_size,
                latent_mlp_hidden_size,
                self._z1_size)
        # z^2_{t+1} ~ p(z^2_{t+1} | z^1_{t+1}, z^2_t, a_t)
        self._z2_prior = MultivariateNormalDiag(
                self._z1_size + self._z2_size + self._action_size,
                latent_mlp_hidden_size,
                self._z2_size)
        # r_t ~ p(r_t | z_t, a_t, z_{t+1})
        self._reward_model = MultivariateNormalDiag(
                2 * self._full_latent_size + self._action_size,
                latent_mlp_hidden_size,
                1,
                scale=reward_stddev)
        # q(z^1_1 | x_1)
        self._z1_first_posterior = MultivariateNormalDiag(
                self._feature_size,
                latent_mlp_hidden_size,
                self._z1_size)
        self._z2_first_posterior = self._z2_first_prior
        # q(z^1_{t+1} | x_{t+1},z^2_t,a_t)
        self._z1_posterior = MultivariateNormalDiag(
                self._feature_size + self._z2_size + self._action_size,
                latent_mlp_hidden_size,
                self._z1_size)
        self._z2_posterior = self._z2_prior

    def forward(self, *experience):
        images, actions, rewards, step_types = experience
        features = self.feature_extractor(images)

        z1_posterior_samples, z1_posterior_dists, z2_posterior_samples, z2_posterior_dists = \
            self.posterior_samples_and_dists(features, actions, step_types)

        z1_samples = torch.stack(z1_posterior_samples, axis=1)
        z2_samples = torch.stack(z2_posterior_samples, axis=1)

        return z1_samples, z2_samples

    def compute_loss(self, experience):
        images, actions, rewards, step_types = experience

        features = self.feature_extractor(images)

        z1_posterior_samples, z1_posterior_dists, z2_posterior_samples, z2_posterior_dists = \
            self.posterior_samples_and_dists(features, actions, step_types)

        z1_samples = torch.stack(z1_posterior_samples, axis=1)
        z2_samples = torch.stack(z2_posterior_samples, axis=1)

        z_samples = torch.cat([z1_samples, z2_samples], axis=-1)

        z1_prior_dists = self.z1_prior_dists_given_samples(
                z1_samples,
                z2_samples,
                actions,
                step_types)

        reconstruction_dist = self.decoder(z_samples)

        reward_input = torch.cat([
            z1_samples[:,:-1],
            z2_samples[:,:-1],
            actions[:,:],
            z1_samples[:,1:],
            z2_samples[:,1:]], axis=-1)
        reward_dist = self._reward_model(reward_input)

        reconstruction_logprobs = reconstruction_dist.log_prob(images).sum(dim=-1)
        reconstruction_loss = -reconstruction_logprobs.mean()

        kl = torch.FloatTensor([0])
        for (p, q) in zip(z1_prior_dists, z1_posterior_dists):
            kl += dtb.kl.kl_divergence(q, p).mean()

        reward_logprobs = reward_dist.log_prob(rewards.unsqueeze(dim=-1))
        reward_loss = -reward_logprobs.mean()

        loss = reconstruction_loss + kl + reward_loss

        artifacts = {
            'z1_posterior_samples': z1_samples,
            'z2_posterior_samples': z2_samples,
            'images': images,
            'features': features,
            'posterior_images': reconstruction_dist.mean,
            'reward_loss': reward_loss,
        }

        return loss, artifacts

    def posterior_samples_and_dists(self, features, actions, step_types):
        # For now, we assume episodes never end in the middle of a batch
        del(step_types)

        z1_samples = []
        z2_samples = []
        z1_dists = []
        z2_dists = []

        # Swap batch and time axes
        features = features.permute(1,0,2)
        actions = actions.permute(1,0,2)

        sequence_length = actions.shape[0]
        first_z1_dist = self._z1_first_posterior(features[0])
        first_z1_sample = first_z1_dist.sample()
        first_z2_dist = self._z2_first_posterior(first_z1_sample)
        first_z2_sample = first_z2_dist.sample()
        z1_samples.append(first_z1_sample)
        z2_samples.append(first_z2_sample)
        z1_dists.append(first_z1_dist)
        z2_dists.append(first_z2_dist)
        for t in range(1, sequence_length + 1):
            z1_input = torch.cat([features[t], z2_samples[t-1], actions[t-1]], axis=-1)
            z1_dist = self._z1_posterior(z1_input)
            z1_sample = z1_dist.sample()
            z2_input = torch.cat([z1_sample, z2_samples[t-1], actions[t-1]], axis=-1)
            z2_dist = self._z2_posterior(z2_input)
            z2_sample = z2_dist.sample()
            z1_samples.append(z1_sample)
            z2_samples.append(z2_sample)
            z1_dists.append(z1_dist)
            z2_dists.append(z2_dist)

        return z1_samples, z1_dists, z2_samples, z2_dists

    def z1_prior_dists_given_samples(self, z1_samples, z2_samples, actions, step_types):
        # For now, we assume episodes never end in the middle of a batch
        del(step_types)

        z1_dists = []

        first_z1_dist = self._z1_first_prior(actions)
        z1_dists.append(first_z1_dist)
        # Swap batch and time axes
        actions = actions.permute(1,0,2)
        z1_samples = z1_samples.permute(1,0,2)
        z2_samples = z2_samples.permute(1,0,2)
        sequence_length = actions.shape[0]
        for t in range(1, sequence_length + 1):
            z1_input = torch.cat([z2_samples[t-1], actions[t-1]], axis=-1)
            z1_dist = self._z1_prior(z1_input)
            z1_dists.append(z1_dist)

        return z1_dists
