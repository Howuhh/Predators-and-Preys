import os
from numpy.core.numeric import roll
import torch
import numpy as np

from utils import ReplayBuffer
from maddpg import MADDPG
from wrapper import VectorizeWrapper

from predators_and_preys_env.env import PredatorsAndPreysEnv
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def rollout(env, agents, greedy=False):    
    (state, global_state), done = env.reset(), False
    total_reward = []
    
    # steps = 0.0
    while not done:
        actions = np.hstack([agent.act(agent_state, greedy=greedy) for agent, agent_state in zip(agents, state)])
        # print(steps)
        # print(predator_actions, prey_actions)
        # print(state_dict)
        # steps += 1
        # print(predator_actions, prey_actions)
        
        state, rewards, done, state_dict = env.step(actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)


def eval_maddpg(maddpg, episodes=10, seed=42):
    env = VectorizeWrapper(PredatorsAndPreysEnv(render=False))
    set_seed(env, seed)

    rewards = [rollout(env, maddpg.agents, greedy=True) for _ in range(episodes)]
        
    return np.vstack(rewards).mean(axis=0)


def train_maddpg(agents_configs, timesteps, buffer_size, batch_size, updates_per_iter, update_every, eval_every, device="cpu", seed=10):
    buffer = ReplayBuffer(size=buffer_size, n_agents=3)
    maddpg = MADDPG(agents_configs, device=device)
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(render=False))
    set_seed(env, seed=seed)
    
    (state, global_state), done = env.reset(), False
    
    for step in range(1, timesteps + 1):
        if done:
            (state, global_state), done = env.reset(), False 
        
        actions = np.hstack([agent.act(agent_state) for agent, agent_state in zip(maddpg.agents, state)])

        next_state, reward, done, next_global_state = env.step(actions)
        buffer.add(state, global_state, actions, reward, next_state, next_global_state, done)
        
        state = next_state
        global_state = next_global_state
        
        if step % update_every == 0 and step > batch_size:
            for _ in range(updates_per_iter):
                batch = buffer.sample(batch_size)
                losses = maddpg.update(batch)
        
            if step % eval_every == 0:
                rewards = eval_maddpg(maddpg, episodes=10, seed=42)
                maddpg.save(f"maddpg.pt")
        
                print(step, rewards, losses)
                
    return maddpg


if __name__ == "__main__":
    from configs import predator_agent_config, prey_agent_config

    # maddpg = train_maddpg(
    #     agents_configs=[predator_agent_config, predator_agent_config] + [prey_agent_config],
    #     timesteps=500_000,
    #     buffer_size=1_000_000,
    #     batch_size=256,
    #     updates_per_iter=25,
    #     update_every=50,
    #     eval_every=1000
    # )

    maddpg = torch.load("maddpg.pt", map_location="cpu")
    env = VectorizeWrapper(PredatorsAndPreysEnv(render=True))
    
    for _ in range(200):
        rollout(env, maddpg.agents, greedy=True)