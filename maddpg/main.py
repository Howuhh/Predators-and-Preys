import os
import shutil
import torch
import numpy as np

from tqdm import tqdm
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


def eval_maddpg(env_config, maddpg, episodes=10, seed=42):
    env = VectorizeWrapper(
        PredatorsAndPreysEnv(config=env_config, render=False), 
        return_state_dict=True
    )
    set_seed(env, seed)

    rewards = [rollout(env, maddpg.agents[0], maddpg.agents[1], greedy=True) for _ in range(episodes)]
        
    return np.vstack(rewards).mean(axis=0)


def train_maddpg(env_config, agents_configs, buffer_config, timesteps, batch_size, 
                 updates_per_iter, update_every, eval_every, device="cpu", seed=10):
    if os.path.exists("agents"):
        shutil.rmtree("agents")
    os.makedirs("agents")
    
    buffer = ReplayBuffer(**buffer_config, device=device)
    maddpg = MADDPG(agents_configs, device=device)
    
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=env_config, render=False))
    set_seed(env, seed=seed)
    
    state, done = env.reset(), False
    
    for step in tqdm(range(1, timesteps + 1)):
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
                rewards = eval_maddpg(env_config, maddpg, episodes=25, seed=42)
                maddpg.save(f"agents/maddpg_{step}.pt")
        
                print("==" * 15 + f"Step {step}" + "==" * 15)
                for i in range(len(maddpg.agents)):
                    actor_loss, critic_loss = losses[i]
                    print(f"Agent{i + 1} -- Reward: {rewards[i]}, Critic loss: {round(actor_loss, 5)}, Actor loss: {round(critic_loss, 5)}")
        
        for agent in maddpg.agents:
            agent.act_noise = 2.0 - (2.0 - 0.1) * step / timesteps
        # limits scheduling
        # limit = 3 - (3 - 6) * step / timesteps

        # env.env.game.x_limit = limit
        # env.env.game.y_limit = limit
                    
    return maddpg


if __name__ == "__main__":
    from configs import predator_agent_config, prey_agent_config, buffer_config
    from configs import SIMPLE1v1, SIMPLE2v1, SIMPLE2v2, SIMPLE1v2
    
    # maddpg = train_maddpg(
    #     env_config=SIMPLE2v1,
    #     agents_configs=[predator_agent_config, prey_agent_config],
    #     buffer_config=buffer_config,
    #     timesteps=1_000_000,
    #     batch_size=256,
    #     updates_per_iter=1,
    #     update_every=1,
    #     eval_every=20_000,
    #     seed=42
    # )
    maddpg = torch.load("agents/maddpg_100000.pt", map_location="cpu")
    predator, prey = maddpg.agents
    
    SIMPLE2v1["environment"]["time_limit"] = 100
    # SIMPLE2v2["game"]["x_limit"] = 4
    # SIMPLE2v2["game"]["y_limit"] = 4
    env = VectorizeWrapper(PredatorsAndPreysEnv(config=SIMPLE2v1, render=True), return_state_dict=True)
    
    for _ in range(200):
        set_seed(env, np.random.randint(5000, 10000))
        rollout(env, predator_agent=predator, prey_agent=prey, greedy=True)