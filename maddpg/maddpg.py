import torch
import torch.nn.functional as F

import numpy as np

from utils import ReplayBuffer
from agent import Agent


class MADDPG:
    def __init__(self, agents_configs, device):
        self.agents = [Agent(**agent_config, device=device) for agent_config in agents_configs]
        self.device = device
    
    def __to_torch_tensor(self, global_state, agents_actions, agents_rewards, global_next_state, done):
        global_state = torch.FloatTensor(global_state).to(self.device)
        agents_actions = [torch.FloatTensor(agent_actions).to(self.device) for agent_actions in agents_actions]
        agents_rewards = [torch.FloatTensor(agent_rewards).to(self.device).unsqueeze(1) for agent_rewards in agents_rewards]
        global_next_state = torch.FloatTensor(global_next_state).to(self.device)
        done = torch.IntTensor(done).to(self.device).unsqueeze(1)
        
        return global_state, agents_actions, agents_rewards, global_next_state, done
    
    def update(self, batch):
        # for now only works when agents state == global env state, as in PredatorsAndPreys env
        # if you want to fix this - change ReplayBuffer sampling and this function (not too hard tho)
        global_state, agents_actions, agents_rewards, global_next_state, done = self.__to_torch_tensor(*batch)
        
        agents_actions = torch.cat(agents_actions, dim=-1)
        # # target agents actions for next states
        target_agents_next_actions = torch.cat(
            [torch.FloatTensor(agent.target_actor(global_next_state)).to(self.device) for agent in self.agents], dim=-1
        ).detach()
        # # current actors actions for updates
        # new_agents_actions = torch.cat(
        #     [torch.FloatTensor(agent.actor(global_state)).to(self.device) for agent in self.agents], dim=-1
        # )
        losses = []

        # agents updates
        for agent_idx, agent in enumerate(self.agents):
            # critic update
            Q_next = agent.target_critic(global_next_state, target_agents_next_actions)
            Q_target = agents_rewards[agent_idx] + (1 - done) * agent.gamma * Q_next
        
            Q = agent.critic(global_state, agents_actions)
            
            assert Q.shape == Q_target.shape
            
            critic_loss = F.mse_loss(Q, Q_target.detach())
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # current actors actions for updates
            new_agents_actions = torch.cat(
                [torch.FloatTensor(agent.actor(global_state)).to(self.device) for agent in self.agents], dim=-1
            )
            
            # actor update
            actor_loss = -agent.critic(global_state, new_agents_actions).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # target networks update
            agent.update_target_networks()
            
            losses.append([critic_loss.item(), actor_loss.item()])
            
        return losses     
            
    def save(self, path):
        torch.save(self, path)

