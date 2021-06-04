import torch
import numpy as np

from wrapper import VectorizeWrapper
from predators_and_preys_env.env import PredatorsAndPreysEnv

env = VectorizeWrapper(PredatorsAndPreysEnv(render=True))

done = True
step_count = 0
while True:
    if done:
        print("reset")
        state, global_state = env.reset()
        step_count = 0
        
    actions = np.zeros(env.predator_action_size + env.prey_action_size)
        
    state, reward, done, global_state = env.step(actions)
    step_count += 1

    print(state)

    print(f"step {step_count}")