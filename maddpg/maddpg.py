import torch
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
from agent import Agent


class MADDPG:
    def __init__(self, agents_configs, device="cpu"):
        self.agents = [Agent(**agent_config, device=device) for agent_config in agents_configs]
        self.device = device
        
    def update(self, batch, step):
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
            Q_next = torch.minimum(
                agent.target_critic1(global_next_state, target_agents_next_actions),
                agent.target_critic2(global_next_state, target_agents_next_actions)
            )
            Q_target = agents_rewards[agent_idx] + (1 - done) * agent.gamma * Q_next
        
            Q1 = agent.critic1(global_state, agents_actions)
            Q2 = agent.critic2(global_state, agents_actions)
            
            assert Q1.shape == Q2.shape == Q_target.shape
            
            critic1_loss = F.mse_loss(Q1, Q_target.detach())
            critic2_loss = F.mse_loss(Q2, Q_target.detach())
            
            critic_loss = critic1_loss + critic2_loss
                        
            agent.critic1_optimizer.zero_grad()
            agent.critic2_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic1_optimizer.step()
            agent.critic2_optimizer.step()
            
            if step % 50 == 0: # 50 best
            # current actors actions for updates
                with torch.no_grad():
                    new_agents_actions = deepcopy(agents_actions)
                    offset = sum([self.agents[past_idx].actor.action_size for past_idx in range(agent_idx)])
                    
                raw_actions, constrained_actions = agent.actor(global_state, return_raw=True)
                new_agents_actions[:, offset:offset+agent.actor.action_size] = constrained_actions
                
                # actor update
                actor_loss = -agent.critic1(global_state, new_agents_actions).mean()
                actor_loss += agent.actions_decay * (raw_actions**2).mean()
                
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                
                # target networks update
                agent.update_target_networks()
        
                if step % 20_000 == 0:
                    print(f"Agent{agent_idx + 1} -- Critic loss: {round(critic_loss.item(), 5)}, Actor loss: {round(actor_loss.item(), 5)}")
        
            # losses.append([critic1_loss.item(), actor_loss.item()])
            
        return losses     
            
    def save(self, path):
        torch.save(self, path)

