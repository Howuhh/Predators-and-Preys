import torch
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
from agent import Agent


class MADDPG:
    def __init__(self, agents_configs, device):
        self.agents = [Agent(**agent_config, device=device) for agent_config in agents_configs]
        self.device = device
        
    def update(self, batch):
        global_state, agents_actions, agents_rewards, global_next_state, done = batch
                
        agents_actions = torch.cat(agents_actions, dim=-1)
        # target agents actions for next states
        target_agents_next_actions = torch.cat(
            [agent.target_actor(global_next_state).to(self.device) for agent in self.agents], dim=-1
        ).detach()

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
            with torch.no_grad():
                new_agents_actions = deepcopy(agents_actions)
                offset = sum([self.agents[past_idx].actor.action_size for past_idx in range(agent_idx)])
                
            new_agents_actions[:, offset:offset+agent.actor.action_size] = agent.actor(global_state).to(self.device)
            
            # actor update
            actor_loss = -agent.critic(global_state, new_agents_actions).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # target networks update
            agent.update_target_networks()
            
            if critic_loss.item() == np.inf:
                print(critic_loss, critic_loss.item() == np.inf)
                print(agents_rewards[agent_idx])
            
            losses.append([critic_loss.item(), actor_loss.item()])
            
        return losses     
            
    def save(self, path):
        torch.save(self, path)

