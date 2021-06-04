
OBSTACLES_STATE_DIM = 3
OBSTACLES_TEAM_SIZE = 0

PREY_STATE_DIM = 5
PREY_TEAM_SIZE = 1

PREDATOR_STATE_DIM = 4
PREDATOR_TEAM_SIZE = 2

STATE_SIZE = PREDATOR_STATE_DIM * PREDATOR_TEAM_SIZE + PREY_STATE_DIM * PREY_TEAM_SIZE + OBSTACLES_TEAM_SIZE * OBSTACLES_STATE_DIM
CRITIC_ACTION_SIZE = PREY_TEAM_SIZE + PREDATOR_TEAM_SIZE


prey_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": 1,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-4, 
    "critic_lr": 1e-4, 
    "tau": 1e-3, 
    "gamma": 0.95, 
    "act_noise": 0.1
}

predator_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": 1,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-4, 
    "critic_lr": 1e-4, 
    "tau": 1e-3, 
    "gamma": 0.95, 
    "act_noise": 0.1
}