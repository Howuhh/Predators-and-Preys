import sys

sys.path.append("../../")

import time
import numpy as np

from predators_and_preys_env.env import PredatorsAndPreysEnv
from examples.simple_chasing_agents.agents import ChasingPredatorAgent, FleeingPreyAgent

env = PredatorsAndPreysEnv(render=True)
predator_agent = ChasingPredatorAgent()
prey_agent = FleeingPreyAgent()

done = True
step_count = 0
state_dict = None
while True:
    if done:
        state_dict = env.reset()
        step_count = 0

    state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
    step_count += 1
