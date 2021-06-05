import os
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
    

def rollout(env, predator_agent, prey_agent, greedy=False):    
    (state, state_dict), done = env.reset(), False
    total_reward = []
    
    steps = 0.0
    while not done:
        if isinstance(predator_agent, ChasingPredatorAgent):
            predator_actions = predator_agent.act(state_dict)
        else:
            predator_actions = predator_agent.act(state, greedy=greedy)
            
        if isinstance(prey_agent, FleeingPreyAgent):
            prey_actions = prey_agent.act(state_dict)
        else:
            prey_actions = prey_agent.act(state, greedy=greedy)

        steps += 1
        
        state, rewards, done, state_dict = env.step(predator_actions, prey_actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)


def eval_maddpg(maddpg, episodes=10, seed=42):
    env = VectorizeWrapper(PredatorsAndPreysEnv(render=False), return_state_dict=True)
    set_seed(env, seed)

    rewards = [rollout(env, maddpg.agents[0], maddpg.agents[1], greedy=True) for _ in range(episodes)]
        
    return np.vstack(rewards).mean(axis=0)


def train_maddpg(agents_configs, buffer_config, timesteps, batch_size, 
                 updates_per_iter, update_every, eval_every, device="cpu", seed=10):
    buffer = ReplayBuffer(**buffer_config, device=device)
    maddpg = MADDPG(agents_configs, device=device)
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(render=False))
    set_seed(env, seed=seed)
    
    state, done = env.reset(), False
    
    for step in range(1, timesteps + 1):
        if done:
            state, done = env.reset(), False 
        
        actions = [agent.act(state) for agent in maddpg.agents]
        
        next_state, reward, done = env.step(*actions)
        buffer.add(state, actions, reward, next_state, done)

        state = next_state
        
        if step % update_every == 0 and step > batch_size:
            for _ in range(updates_per_iter):
                batch = buffer.sample(batch_size)
                losses = maddpg.update(batch)
        
            if step % eval_every == 0:
                rewards = eval_maddpg(maddpg, episodes=25, seed=42)
                maddpg.save(f"maddpg.pt")
        
                print("==" * 20 + f"Step {step}" + "==" * 20)
                for i in range(len(maddpg.agents)):
                    actor_loss, critic_loss = losses[i]
                    print(f"Agent{i + 1} -- Reward: {rewards[i]}, Critic loss: {round(actor_loss, 5)}, Actor loss: {round(critic_loss, 5)}")
                
    return maddpg


if __name__ == "__main__":
    from configs import predator_agent_config, prey_agent_config, buffer_config
    
    maddpg = train_maddpg(
        agents_configs=[predator_agent_config, prey_agent_config],
        buffer_config=buffer_config,
        timesteps=100_000,
        batch_size=256,
        updates_per_iter=1,
        update_every=1,
        eval_every=10
    )

    # baseline_prey = FleeingPreyAgent()
    # baseline_predator = ChasingPredatorAgent()
    
    # maddpg = torch.load("maddpg.pt", map_location="cpu")
    # predator, prey = maddpg.agents
    
    # env = VectorizeWrapper(PredatorsAndPreysEnv(render=True), return_state_dict=True)
    
    # for _ in range(200):
    #     rollout(env, predator_agent=predator, prey_agent=prey, greedy=True)