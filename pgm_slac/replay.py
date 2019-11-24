import numpy as np
import torch

class SequenceReplayBuffer(object):
    def __init__(self, capacity, sequence_length, image_shape, action_shape):
        self._pointer = 0
        self._image_buf = np.zeros(shape=(capacity, sequence_length, *image_shape))
        self._action_buf = np.zeros(shape=(capacity, sequence_length, *action_shape))
        self._rew_buf = np.zeros(shape=(capacity, sequence_length))
        self._step_type_buf = np.zeros(shape=(capacity, sequence_length))
        self._capacity = capacity
        self._weight = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, image_seq, action_seq, rew_seq, step_type):
        if self._weight < self._capacity:
            self._weight += 1
        else:
            self._image_buf[self._pointer] = image_seq
            self._action_buf[self._pointer] = action_seq
            self._rew_buf[self._pointer] = rew_seq
            self._step_type_bug[self._pointer] = step_type
        self._pointer = (self._pointer + 1) % self._capacity

    def sample(self, batch_size=1):
        indices = np.random.randint(0, self._weight, size=batch_size)
        return (
            torch.FloatTensor(self._image_buf[indices]).to(self._device),
            torch.FloatTensor(self._action_buf[indices]).to(self._device),
            torch.FloatTensor(self._rew_buf[indices]).to(self._device),
            self._step_type_buf[indices]
        )
