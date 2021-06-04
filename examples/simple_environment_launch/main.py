# import sys

# sys.path.append("../../")

import numpy as np

from predators_and_preys_env.env import PredatorsAndPreysEnv

env = PredatorsAndPreysEnv(render=True)

done = True
step_count = 0
while True:
    if done:
        print("reset")
        state = env.reset()
        step_count = 0
        
    state, reward, done = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
    step_count += 1

    print(reward)

    print(f"step {step_count}")
