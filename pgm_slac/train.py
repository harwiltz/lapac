import argparse
import gym
import numpy as np
import os
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from pgm_slac.environments import custom_environments
from pgm_slac.pgm_slac_agent import PGMSlacAgent
from pgm_slac.replay import SequenceReplayBuffer
from pgm_slac.trajectory import StepType

def train(
        log_dir,
        experiment_name,
        env_universe='custom',
        env_name='CarRacing-v0',
        sequence_length=4,
        feature_size=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        ent_lr=3e-4,
        model_lr=1e-4,
        initial_collect_time=100,
        replay_buffer_capacity=10000,
        max_timesteps=100000,
        max_episode_length=500,
        steps_per_epoch=10000,
        batch_size=32):

    experiment_dir = os.path.join(log_dir, experiment_name)
    writer = SummaryWriter(experiment_dir)

    env, obs_space, action_space = load_environment(env_universe, env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = SequenceReplayBuffer(
            replay_buffer_capacity,
            sequence_length,
            obs_space.shape,
            action_space.shape)

    agent = PGMSlacAgent(
            obs_space,
            action_space,
            sequence_length=sequence_length,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            ent_lr=ent_lr,
            model_lr=model_lr)

    img = env.reset()
    img_ctx, action_ctx, reward_ctx, step_types = agent.clear_context()
    step_type = StepType.first

    episode_reward = 0
    cur_episode_length = 0
    last_episode_reward = 0
    last_episode_length = 0
    for t in range(max_timesteps):
        cur_episode_length += 1
        img_ctx = np.concatenate([img_ctx[1:], np.expand_dims(img, 0)])
        replay.add(img_ctx, action_ctx, reward_ctx, step_types)
        if t < initial_collect_time:
            action = env.action_space.sample()
        else:
            action = agent.action((img_ctx, action_ctx, step_types)).numpy()
        img, rew, done, info = env.step(action)
        episode_reward += rew
        action_ctx = np.concatenate([action_ctx[1:], np.expand_dims(action, 0)])
        reward_ctx = np.concatenate([reward_ctx[1:], np.array([rew])])
        step_types = step_types[1:] + [step_type]
        done = done or (cur_episode_length >= max_episode_length)
        if not done:
            step_type = StepType.mid
        else:
            step_type = StepType.first
            img = env.reset()
            img_ctx, action_ctx, reward_ctx, step_types = agent.clear_context()
            last_episode_reward = episode_reward
            last_episode_length = cur_episode_length
            episode_reward = 0
            cur_episode_length = 0
        if t > initial_collect_time:
            experience = replay.sample(batch_size)
            artifacts = agent.train(experience)
            artifacts.update({
                'done': done,
                'episode_length': last_episode_length,
                'episode_reward': last_episode_reward,
            })
            update_summaries(writer, artifacts, t)

def update_summaries(writer, artifacts, t, image_freq=50):
    writer.add_scalar('Model/Model Loss', artifacts['model_loss'], t)
    writer.add_scalar('Actor/Actor Loss', artifacts['actor_loss'], t)
    writer.add_scalar('Actor/Log Policy', artifacts['actor_log_pi'], t)
    writer.add_scalar('Critic/Critic Loss', artifacts['critic_loss'], t)
    writer.add_scalar('Entropy/Alpha Loss', artifacts['ent_loss'], t)
    writer.add_scalar('Entropy/Alpha', artifacts['ent'], t)
    writer.add_scalar('Performance/Episode Length', artifacts['episode_length'], t)
    writer.add_scalar('Performance/Episode Score', artifacts['episode_reward'], t)
    if t % image_freq == 0:
        writer.add_image('Ground Truth', create_image_chain(artifacts['images']))
        writer.add_image('Posterior Images', create_image_chain(artifacts['posterior_images']))

def create_image_chain(images, chain_length=4):
    # images shape: (Batch, W, H, 3)
    images = images[:chain_length, -1, :, :].permute(0, 3, 2, 1)
    return torchvision.utils.make_grid(images, nrow=chain_length)

def load_environment(env_universe, env_name):
    if env_universe == 'custom':
        env = custom_environments.load(env_name)
        return env, env.custom_observation_space(), env.action_space
    else:
        env = gym.make(env_name)
        return env, env.observation_space, env.action_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pgm-slac agent')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--experiment_name', type=str, default='pgm-slac')
    args = parser.parse_args()
    train(args.log_dir, args.experiment_name)
