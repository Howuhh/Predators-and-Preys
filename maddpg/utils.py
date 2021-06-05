import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, size, n_agents, state_size, action_sizes, device="cpu"):
        self.n_agents = n_agents
        self.device = device
        
        self.size = size
        self._buffer_idx = 0
        self._consumed = 0
        
        # global_state, next_global_state, done
        self.global_states = torch.empty((size, state_size), dtype=torch.float, device=device)
        self.global_next_states = torch.empty((size, state_size), dtype=torch.float, device=device)
        self.done = torch.empty((size, 1), dtype=torch.int, device=device)
        
        # state, action, reward, next_state
        self.agents_actions = [torch.empty((size, action_sizes[i]), dtype=torch.float, device=device) for i in range(n_agents)]
        self.agents_rewards = [torch.empty((size, 1), dtype=torch.float, device=device) for _ in range(n_agents)]

    def add(self, global_state, agents_actions, agents_rewards, global_next_state, done):
        # add to global buffer (for critic)
        self.global_states[self._buffer_idx] = torch.tensor(global_state, device=self.device)
        self.global_next_states[self._buffer_idx] = torch.tensor(global_next_state, device=self.device)
        self.done[self._buffer_idx] = torch.tensor(done, device=self.device)
        
        # add to agents buffers
        for agent_idx in range(self.n_agents):
            self.agents_actions[agent_idx][self._buffer_idx] = torch.tensor(agents_actions[agent_idx], device=self.device)
            self.agents_rewards[agent_idx][self._buffer_idx] = torch.tensor(agents_rewards[agent_idx], device=self.device)

        self._buffer_idx = (self._buffer_idx + 1) % self.size
        self._consumed = self._consumed + 1


    def sample(self, batch_size):
        real_size = min(self.size, self._consumed)
        
        assert real_size >= batch_size, "buffer is smaller than batch"
        
        idxs = np.random.choice(real_size, batch_size, replace=False)
        
        # batch sampling
        global_state = self.global_states[idxs]
        global_next_state = self.global_next_states[idxs]
        done = self.done[idxs]
        
        agents_actions = [agent_actions[idxs] for agent_actions in self.agents_actions]
        agents_rewards = [agent_rewards[idxs] for agent_rewards in self.agents_rewards]
                
        return global_state, agents_actions, agents_rewards, global_next_state, done