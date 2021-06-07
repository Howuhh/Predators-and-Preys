import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

# class StateEncoder(nn.Module):
#     def __init__(self, team_size, obstacle_size, embedding_size=16):
#         super().__init__()
#         self.team_size = team_size
        
#         self.team_fc = nn.Sequential(
#             nn.Linear(team_size, embedding_size),
#             nn.Tanh()
#         )
#         self.obst_fc = nn.Sequential(
#             nn.Linear(obstacle_size, embedding_size),
#             nn.Sigmoid()
#         )
        
#     def forward(self, state):
#         team, obst = torch.split(state, self.team_size, dim=-1)
        
#         team_emb = self.team_fc(team)
#         obst_emb = self.obst_fc(obst)

#         return torch.cat([team_emb, obst_emb], dim=-1)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.action_size = action_size
        
        self.team_fc = nn.Sequential(
            nn.Linear(4 + 3 * 2, 16),
            nn.Tanh()
        )
        self.obst_fc = nn.Sequential(
            nn.Linear(3 * 4, 16),
            nn.Sigmoid()
        )
        
        self.model = nn.Sequential(
            nn.Linear(32, hidden_size),
            # nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        team, obst = torch.split(state, [3 * 2 + 4, 3 * 4], dim=-1)
            
        team_emb = self.team_fc(team)
        obst_emb = self.obst_fc(obst)

        out = self.model(torch.cat([team_emb, obst_emb], dim=-1)) # лучший способ
        # out = self.model(team_emb * obst_emb)
        # out = self.model(state)
        
        return torch.tanh(out / 30)


# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=64):
#         super().__init__()
#         self.action_size = action_size
        
#         self.model = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, action_size),
#         )
#         self.model[-1].weight.data.uniform_(-3e-3, 3e-3)
        
#     def forward(self, state):
#         out = self.model(state)

#         return torch.tanh(out / 30)
        

class CentralizedCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        
        self.team_fc = nn.Sequential(
            nn.Linear(4 + 3 * 2, 16),
            nn.Tanh()
        )
        self.obst_fc = nn.Sequential(
            nn.Linear(3 * 4, 16),
            nn.Sigmoid()
        )
        
        self.model = nn.Sequential(
            # nn.Linear(state_size + action_size, hidden_size),
            nn.Linear(32 + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        team, obst = torch.split(state, [4 + 3 * 2, 3 * 4], dim=-1)
        
        team_emb = self.team_fc(team)
        obst_emb = self.obst_fc(obst)
        
        state_action = torch.cat([team_emb, obst_emb, action], dim=-1)
        # state_action = torch.cat([state, action], dim=-1)
        
        return self.model(state_action)


class Agent:
    def __init__(self, state_size, actor_action_size, critic_action_size, actor_hidden_size=64, critic_hidden_size=64, 
                 actor_lr=1e-3, critic_lr=1e-3, tau=1e-3, gamma=0.99, act_noise=0.1, device="cpu"):
        self.actor = Actor(state_size, actor_action_size, actor_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = CentralizedCritic(state_size, critic_action_size, critic_hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        with torch.no_grad():
            self.target_actor = deepcopy(self.actor)
            self.target_critic = deepcopy(self.critic)    
        
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.act_noise = act_noise
        
    def __soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
                
    def act(self, state, greedy=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = self.actor(state).cpu().numpy()
            
        if not greedy:
            action = np.clip(action + self.act_noise * np.random.randn(*action.shape), -1.0, 1.0)
            
        return action
                       
    def update_target_networks(self):
        with torch.no_grad():
            self.__soft_update(self.target_actor, self.actor)
            self.__soft_update(self.target_critic, self.critic)