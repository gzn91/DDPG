import numpy as np
import random


class Memory:

    def __init__(self,limit, dim=0):
        self.data = np.zeros((limit, dim), dtype='float32')
        self.limit = limit
        self.size = 0
        self.idx = 0

    def __len__(self):
        return self.size

    def append(self,data):
        if self.size < self.limit:
            self.data[self.size] = data
            self.size += 1
        else:
            self.data[(self.size + self.idx) % self.limit] = data
            self.idx = (self.idx + 1) % self.limit

    def sample(self, batch_idxs):
        return self.data[(self.idx + batch_idxs) % self.limit]


class ReplayBuffer:

    def __init__(self, buffer_size, state_dim=1, action_dim=1, random_seed=1337):
        self.states0 = Memory(buffer_size, dim=state_dim)
        self.actions = Memory(buffer_size, dim=action_dim)
        self.rewards = Memory(buffer_size, dim=1)
        self.terminals = Memory(buffer_size, dim=1)
        self.states1 = Memory(buffer_size, dim=state_dim)
        random.seed(random_seed)

    def add(self, state0, action, reward, terminal1, state1):

        self.states0.append(state0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states1.append(state1)
        self.terminals.append(terminal1)

    @property
    def size(self):
        return len(self.states0)

    def sample_batch(self, batch_size):

        batch_idxs = np.array(random.sample(range(self.size - 2), k=batch_size))

        state0_batch = self.states0.sample(batch_idxs)
        state1_batch = self.states1.sample(batch_idxs)
        action_batch = self.actions.sample(batch_idxs)
        reward_batch = self.rewards.sample(batch_idxs)
        terminal_batch = self.terminals.sample(batch_idxs)

        return state0_batch, action_batch, reward_batch, terminal_batch, state1_batch
