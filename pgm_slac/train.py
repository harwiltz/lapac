import gym
import numpy as np
import torch

from pgm_slac.environments import custom_environments
from pgm_slac.pgm_slac_agent import PGMSlacAgent
from pgm_slac.replay import SequenceReplayBuffer
from pgm_slac.trajectory import StepType

def train(
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
        max_timesteps=1000,
        steps_per_epoch=10000,
        batch_size=32):

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

    for t in range(max_timesteps):
        img_ctx = np.concatenate([img_ctx[1:], np.expand_dims(img, 0)])
        replay.add(img_ctx, action_ctx, reward_ctx, step_types)
        if t < initial_collect_time:
            action = env.action_space.sample()
        else:
            action = agent.action((img_ctx, action_ctx, step_types)).numpy()
        img, rew, done, info = env.step(action)
        action_ctx = np.concatenate([action_ctx[1:], np.expand_dims(action, 0)])
        reward_ctx = np.concatenate([reward_ctx[1:], np.array([rew])])
        step_types = step_types[1:] + [step_type]
        if not done:
            step_type = StepType.mid
        else:
            step_type = StepType.first
            img = env.reset()
            img_ctx, action_ctx, reward_ctx, step_types = agent.clear_context()
        if t > initial_collect_time:
            experience = replay.sample(batch_size)
            agent.train(experience)

def load_environment(env_universe, env_name):
    if env_universe == 'custom':
        env = custom_environments.load(env_name)
        return env, env.custom_observation_space(), env.action_space
    else:
        env = gym.make(env_name)
        return env, env.observation_space, env.action_space

if __name__ == "__main__":
    train()
