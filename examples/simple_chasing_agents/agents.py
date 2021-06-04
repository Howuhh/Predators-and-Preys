import numpy as np
from predators_and_preys_env.agent import PredatorAgent, PreyAgent


def distance(first, second):
    return ((first["x_pos"] - second["x_pos"])**2 + (first["y_pos"] - second["y_pos"])**2)**0.5

class ChasingPredatorAgent(PredatorAgent):
    def act(self, state_dict, **kwargs):
        action = []
        for predator in state_dict["predators"]:
            closest_prey = None
            for prey in state_dict["preys"]:
                if not prey["is_alive"]:
                    continue
                if closest_prey is None:
                    closest_prey = prey
                else:
                    if distance(closest_prey, predator) > distance(prey, predator):
                        closest_prey = prey
            if closest_prey is None:
                action.append(0.)
            else:
                action.append(np.arctan2(closest_prey["y_pos"] - predator["y_pos"],
                                         closest_prey["x_pos"] - predator["x_pos"]) / np.pi)
        return action


class FleeingPreyAgent(PreyAgent):
    def act(self, state_dict, **kwargs):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return action