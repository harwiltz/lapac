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
        replay_buffer_capacity=10000):

    env, obs_space, action_space = load_environment(env_universe, env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = SequenceReplayBuffer(replay_buffer_capacity)

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

    obs = env.reset()

    img_ctx = np.zeros([sequence_length] + list(obs_space.shape))
    img_ctx = np.concatenate([img_ctx, np.expand_dims(obs, 0)])
    action_ctx = np.zeros([sequence_length] + list(action_space.shape))
    reward_ctx = np.zeros(sequence_length)
    step_types = [StepType.first for _ in range(sequence_length)]

    step_type = StepType.first

    for _ in range(initial_collect_time):
        replay.add(img_ctx, action_ctx, reward_ctx, step_types)
        action = agent.action((img_ctx, action_ctx, step_types))
        img, rew, done, info = env.step(action.numpy())
        img_ctx = np.concatenate([img_ctx[1:], np.expand_dims(img, 0)])
        action_ctx = np.concatenate([action_ctx[1:], np.expand_dims(action, 0)])
        reward_ctx = np.concatenate([reward_ctx[1:], np.array([rew])])
        step_types = step_types[1:] + [step_type]
        if not done:
            step_type = StepType.mid
        else:
            step_type = StepType.first

def load_environment(env_universe, env_name):
    if env_universe == 'custom':
        env = custom_environments.load(env_name)
        return env, env.custom_observation_space(), env.action_space
    else:
        env = gym.make(env_name)
        return env, env.observation_space, env.action_space

if __name__ == "__main__":
    train()
