import numpy as np


class VectorizeWrapper:
    def __init__(self, env, return_state_dict=False):
        self.env = env
        self.return_state_dict = return_state_dict
        
        self.predator_action_size = env.predator_action_size
        self.prey_action_size = env.prey_action_size
        
    @staticmethod
    def _vectorize_state(state_dicts):
        
        def _state_to_array(state_dicts_):
            states = []

            for state_dict in state_dicts_:
                state_dict.pop("speed", None)
                
                states.extend(list(state_dict.values()))
            
            return states
        
        return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]
    
    @staticmethod
    def _vectorize_reward(reward_dicts):
        return [reward_dicts["predators"].mean(), reward_dicts["preys"].mean()]
        
        # def _reward_to_array(reward_dicts_):
        #     return sum([d["reward"] for d in reward_dicts_])
                    
        # return [_reward_to_array(reward_dicts["predators"]), _reward_to_array(reward_dicts["preys"])]
    
    def step(self, predator_actions, prey_actions):
        state_dict, reward, done = self.env.step(predator_actions, prey_actions)
        
        if self.return_state_dict:
            return self._vectorize_state(state_dict), self._vectorize_reward(reward), done, state_dict
        
        return self._vectorize_state(state_dict), self._vectorize_reward(reward), done
    
    def reset(self):
        state_dict = self.env.reset()

        if self.return_state_dict:
            return self._vectorize_state(state_dict), state_dict

        return self._vectorize_state(state_dict)
    
    def seed(self, seed):
        self.env.seed(seed)