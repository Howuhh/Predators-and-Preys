import os
import shutil
import torch
import numpy as np

from utils import ReplayBuffer
from maddpg import MADDPG
from wrapper import VectorizeWrapper

from predators_and_preys_env.env import PredatorsAndPreysEnv


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def rollout(env, agents, greedy=False):    
    (state, _), done = env.reset(), False
    total_reward = []
    
    while not done:
        actions = np.hstack([agent.act(agent_state, greedy=greedy) for agent, agent_state in zip(agents, state)])

        state, rewards, done, _ = env.step(actions)
        total_reward.append(rewards)
    
    return np.vstack(total_reward).sum(axis=0)


def eval_maddpg(env_config, maddpg, episodes=10, seed=42):
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=env_config, render=False))
    set_seed(env, seed)

    rewards = [rollout(env, maddpg.agents, greedy=True) for _ in range(episodes)]
        
    return np.vstack(rewards).mean(axis=0)


def train_maddpg(env_config, agents_configs, buffer_config, timesteps, batch_size, updates_per_iter, update_every, eval_every, device="cpu", seed=10):
    if os.path.exists("agents"):
        shutil.rmtree("agents")
    os.makedirs("agents")
    
    buffer = ReplayBuffer(**buffer_config)
    maddpg = MADDPG(agents_configs, device=device)
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=env_config, render=False))
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
                rewards = eval_maddpg(env_config, maddpg, episodes=25, seed=42)
                maddpg.save(f"agents/maddpg_{step}.pt")
        
                print("==" * 15 + f"Step {step}" + "==" * 15)
                for i in range(len(maddpg.agents)):
                    actor_loss, critic_loss = losses[i]
                    print(f"Agent{i + 1} -- Reward: {rewards[i]}, Critic loss: {round(actor_loss, 5)}, Actor loss: {round(critic_loss, 5)}")
                
    return maddpg


if __name__ == "__main__":
    from configs import predator_agent_config, prey_agent_config, buffer_config
    from configs import SIMPLE2v1, SIMPLE2v2

    agents_configs = [predator_agent_config, predator_agent_config] + [prey_agent_config]

    maddpg = train_maddpg(
        env_config=SIMPLE2v1,
        agents_configs=agents_configs,
        buffer_config=buffer_config,
        timesteps=1_000_000,
        batch_size=256,
        updates_per_iter=1,
        update_every=1,
        eval_every=10_000
    )

    # maddpg = torch.load("agents/maddpg_30000.pt", map_location="cpu")
    # env = VectorizeWrapper(PredatorsAndPreysEnv(config=SIMPLE2v1, render=True))
    
    # for _ in range(200):
    #     set_seed(env, np.random.randint(0, 100000))
    #     rollout(env, maddpg.agents, greedy=True)