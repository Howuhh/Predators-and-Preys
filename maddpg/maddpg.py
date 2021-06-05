from copy import deepcopy
import torch
import torch.nn.functional as F

from utils import ReplayBuffer
from agent import Agent


class MADDPG:
    def __init__(self, agents_configs, device):
        self.agents = [Agent(**agent_config, device=device) for agent_config in agents_configs]
        self.device = device
    
    def update(self, batch):
        agents_states, global_state, agents_actions, agents_rewards, agents_next_states, global_next_state, done = batch
        
        agents_actions = torch.cat(agents_actions, dim=-1)
        
        # target agents actions for next states 
        target_agents_next_actions = []
        for agent_idx, agent in enumerate(self.agents):
            target_agents_next_actions.append(
                agent.target_actor(agents_next_states[agent_idx]).to(self.device)
            )    
        target_agents_next_actions = torch.cat(target_agents_next_actions, dim=-1)

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
                current_agents_actions = deepcopy(agents_actions)
            current_agents_actions[:, agent_idx] = agent.actor(agents_states[agent_idx]).to(self.device).squeeze(1)
            
            # actor update
            actor_loss = -agent.critic(global_state, current_agents_actions).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # target networks update
            agent.update_target_networks()
        
            losses.append([critic_loss.item(), actor_loss.item()])
            
        return losses     
            
    def save(self, path):
        torch.save(self, path)

