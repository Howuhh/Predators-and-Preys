import torch
from maddpg import MADDPG
from wrapper import VectorizeWrapper

from predators_and_preys_env.env import PredatorsAndPreysEnv
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent

from main import rollout

baseline_prey = FleeingPreyAgent()
baseline_predator = ChasingPredatorAgent()

maddpg = torch.load("maddpg.pt", map_location="cpu")
predator, prey = maddpg.agents

env = VectorizeWrapper(PredatorsAndPreysEnv(render=True), return_state_dict=True)

for _ in range(20):
    print(rollout(env, predator_agent=predator, prey_agent=prey, greedy=True))