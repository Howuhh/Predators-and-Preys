from copy import deepcopy


class VectorizeWrapper:
    def __init__(self, env, return_state_dict=False):
        self.env = env
        self.return_state_dict = return_state_dict
        
        self.predator_action_size = env.predator_action_size
        self.prey_action_size = env.prey_action_size
        
    @staticmethod
    def _death_masking(state_dicts):
        state_dicts = deepcopy(state_dicts)
        for i, prey in enumerate(state_dicts["preys"]):
            if not prey["is_alive"]:
                prey["x_pos"] = 0
                prey["y_pos"] = 0
                prey["radius"] = 0
                # prey["speed"] = 0
                prey["is_alive"] = i + 2
        # state_dicts["preys"] = sorted(state_dicts["preys"], key=lambda d: d["is_alive"])
        return state_dicts

    @staticmethod
    def _vectorize_state(state_dicts):
        
        def _state_to_array(state_dicts_):
            states = []

            for state_dict in state_dicts_:
                # state_dict.pop("is_alive", None)
                state_dict.pop("speed", None)
                
                states.extend(list(state_dict.values()))
            
            return states
        
        return [*_state_to_array(state_dicts["predators"]), *_state_to_array(state_dicts["preys"]), *_state_to_array(state_dicts["obstacles"])]
    
    @staticmethod
    def _vectorize_reward(reward_dicts):
        return [reward_dicts["predators"].mean(), reward_dicts["preys"].mean()]
        # def _reward_to_array(reward_dicts):
        #             return sum([d["reward"] for d in reward_dicts])

        # return [_reward_to_array(reward_dicts["predators"]), _reward_to_array(reward_dicts["preys"])]
    
    def step(self, predator_actions, prey_actions):
        state_dict, reward, done = self.env.step(predator_actions, prey_actions)
        
        state_dict = self._death_masking(state_dict)
        
        if self.return_state_dict:
            return self._vectorize_state(state_dict), self._vectorize_reward(reward), done, state_dict
        
        return self._vectorize_state(state_dict), self._vectorize_reward(reward), done
    
    def reset(self):
        state_dict = self.env.reset()
        
        state_dict = self._death_masking(state_dict)

        if self.return_state_dict:
            return self._vectorize_state(state_dict), state_dict

        return self._vectorize_state(state_dict)
    
    def seed(self, seed):
        self.env.seed(seed)