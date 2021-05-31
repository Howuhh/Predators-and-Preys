# import sys

# sys.path.append("../../")ls 

from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np

env = PredatorsAndPreysEnv(render=True)

done = True
step_count = 0
while True:
    if done:
        print("reset")
        state = env.reset()
        step_count = 0
    
    state, done = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
    step_count += 1

    print(f"step {step_count}")
