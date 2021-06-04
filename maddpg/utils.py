import numpy as np

from dataclasses import dataclass
from collections import deque

class ReplayBuffer:
    def __init__(self, size, n_agents):
        self.n_agents = n_agents
        self.size = size
        
        self.global_buffer = deque(maxlen=size)
        # state, action, reward, next_state, done
        self.agents_buffer = [deque(maxlen=size) for _ in range(n_agents)]
        
    def add(self, global_state, agent_actions, agent_rewards, global_next_state, done):
        self.global_buffer.append((global_state, global_next_state, done))
                
        for agent_idx in range(self.n_agents):
            self.agents_buffer[agent_idx].append(
                (agent_actions[agent_idx], agent_rewards[agent_idx])
            )

    def sample(self, batch_size):
        assert len(self.global_buffer) <= self.size
        
        idxs = np.random.choice(len(self.global_buffer), batch_size, replace=False)
        
        global_state, global_next_state, done = zip(*[self.global_buffer[idx] for idx in idxs])
        
        agent_actions, agent_rewards = [], []
        
        for agent_idx in range(self.n_agents):
            actions, rewards = zip(*[self.agents_buffer[agent_idx][idx] for idx in idxs])
            
            agent_actions.append(actions)
            agent_rewards.append(rewards)
        
        return global_state, agent_actions, agent_rewards, global_next_state, done
