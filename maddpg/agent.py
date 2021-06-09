import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

class StateEncoder(nn.Module):
    def __init__(self, teams_state_size, obstacle_state_size, embedding_size=16):
        super().__init__()
        self.teams_state_size = teams_state_size
        self.obstacle_state_size = obstacle_state_size
        
        self.team_fc = nn.Sequential(
            nn.Linear(teams_state_size, embedding_size),
            nn.Tanh()
        )
        self.obst_fc = nn.Sequential(
            nn.Linear(obstacle_state_size, embedding_size),
            nn.Sigmoid()
        )
        
    def forward(self, state):
        team, obst = torch.split(state, [self.teams_state_size, self.obstacle_state_size], dim=-1)
        
        team_emb = self.team_fc(team)
        obst_emb = self.obst_fc(obst)

        return torch.cat([team_emb, obst_emb], dim=-1)


class Actor(nn.Module):
    def __init__(self, teams_state_size, obstacle_state_size, action_size, embedding_size=16, hidden_size=64, temperature=1.0):
        super().__init__()
        self.action_size = action_size
        self.temp = temperature
        
        if obstacle_state_size == 0:
            self.model = nn.Sequential(
                nn.Linear(teams_state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
            )
        else:
            self.model = nn.Sequential(
                StateEncoder(
                    teams_state_size=teams_state_size, 
                    obstacle_state_size=obstacle_state_size, 
                    embedding_size=embedding_size
                ),
                nn.Linear(2 * embedding_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
            )
        self.model[-1].weight.data.uniform_(-1e-4, 1e-4)
        
    def forward(self, state):
        out = self.model(state)
        
        return torch.tanh(out / self.temp) # temp=30 best
    

class CentralizedCritic(nn.Module):
    def __init__(self, teams_state_size, obstacle_state_size, action_size, embedding_size=16, hidden_size=64):
        super().__init__()
        self.obstacle_state_size = obstacle_state_size
        state_size = teams_state_size if obstacle_state_size == 0 else 2 * embedding_size
        
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
        )
        
        if obstacle_state_size != 0:
            self.state_encoder = StateEncoder(
                teams_state_size=teams_state_size,
                obstacle_state_size=obstacle_state_size,
                embedding_size=embedding_size
            )
                
    def forward(self, state, action):
        if self.obstacle_state_size != 0:
            state = self.state_encoder(state)
            
        state_action = torch.cat([state, action], dim=-1)
        
        return self.model(state_action)


class Agent:
    def __init__(self, teams_state_size, obstacle_state_size, actor_action_size, 
                 critic_action_size, actor_hidden_size=64, critic_hidden_size=64, 
                 actor_lr=1e-3, critic_lr=1e-3, tau=1e-3, gamma=0.99, act_noise=0.1, device="cpu"):
        self.actor = Actor(teams_state_size, obstacle_state_size, actor_action_size, actor_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = CentralizedCritic(teams_state_size, obstacle_state_size, 
                                        critic_action_size, critic_hidden_size).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = CentralizedCritic(teams_state_size, obstacle_state_size, 
                                         critic_action_size, critic_hidden_size).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        with torch.no_grad():
            self.target_actor = deepcopy(self.actor)
            self.target_critic1 = deepcopy(self.critic1)    
            self.target_critic2 = deepcopy(self.critic2)
        
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
            self.__soft_update(self.target_critic1, self.critic1)
            self.__soft_update(self.target_critic2, self.critic2)