import numpy as np

from dataclasses import dataclass
from collections import deque

class ReplayBuffer:
    def __init__(self, size, n_agents):
        self.n_agents = n_agents
        self.size = size
        
        # global_state, next_global_state, done
        self.global_buffer = deque(maxlen=size)
        # state, action, reward, next_state
        self.agents_buffer = [deque(maxlen=size) for _ in range(n_agents)]
        
    def add(self, agents_states, global_state, agents_actions, agents_rewards, agents_next_states, global_next_state, done):
        self.global_buffer.append((global_state, global_next_state, done))
                
        for agent_idx in range(self.n_agents):
            self.agents_buffer[agent_idx].append(
                (agents_states[agent_idx], agents_actions[agent_idx], agents_rewards[agent_idx], agents_next_states[agent_idx])
            )

    def sample(self, batch_size):
        assert len(self.global_buffer) <= self.size
        
        idxs = np.random.choice(len(self.global_buffer), batch_size, replace=False)
        
        global_state, global_next_state, done = zip(*[self.global_buffer[idx] for idx in idxs])
        
        agents_states, agents_actions, agents_rewards, agents_next_states = [], [], [], []
        
        for agent_idx in range(self.n_agents):
            states, actions, rewards, next_states = zip(*[self.agents_buffer[agent_idx][idx] for idx in idxs])
            
            agents_states.append(states)
            agents_actions.append(actions)
            agents_rewards.append(rewards)
            agents_next_states.append(next_states)
        
        return agents_states, global_state, agents_actions, agents_rewards, agents_next_states, global_next_state, done
