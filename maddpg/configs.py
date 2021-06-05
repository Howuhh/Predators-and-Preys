
OBSTACLES_STATE_DIM = 3
OBSTACLES_TEAM_SIZE = 0

PREY_STATE_DIM = 4
PREY_TEAM_SIZE = 1

PREDATOR_STATE_DIM = 3
PREDATOR_TEAM_SIZE = 2

STATE_SIZE = PREDATOR_STATE_DIM * PREDATOR_TEAM_SIZE + PREY_STATE_DIM * PREY_TEAM_SIZE + OBSTACLES_TEAM_SIZE * OBSTACLES_STATE_DIM
CRITIC_ACTION_SIZE = PREY_TEAM_SIZE + PREDATOR_TEAM_SIZE

buffer_config = {
    "size": 1_000_000,
    "n_agents": 2,
    "state_size": STATE_SIZE,
    "action_sizes": [PREDATOR_TEAM_SIZE, PREY_TEAM_SIZE]
}


prey_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": PREY_TEAM_SIZE,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3, 
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 0.1
}

predator_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": PREDATOR_TEAM_SIZE,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3, 
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 0.3
}