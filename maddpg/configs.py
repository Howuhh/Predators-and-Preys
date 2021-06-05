
OBSTACLES_STATE_DIM = 3
OBSTACLES_TEAM_SIZE = 0

PREY_STATE_DIM = 5
PREY_TEAM_SIZE = 1

PREDATOR_STATE_DIM = 4
PREDATOR_TEAM_SIZE = 2

STATE_SIZE = PREDATOR_STATE_DIM * PREDATOR_TEAM_SIZE + PREY_STATE_DIM * PREY_TEAM_SIZE + OBSTACLES_TEAM_SIZE * OBSTACLES_STATE_DIM
CRITIC_ACTION_SIZE = PREY_TEAM_SIZE + PREDATOR_TEAM_SIZE


buffer_config = {
    "size": 1_000_000,
    "n_agents": PREY_TEAM_SIZE + PREDATOR_TEAM_SIZE,
    "action_size": 1,
    "state_size": STATE_SIZE
}

prey_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": 1,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3, 
    "temperature": 30,
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 0.1
}

predator_agent_config = {
    "state_size": STATE_SIZE,
    "actor_action_size": 1,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3,
    "temperature": 30,
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 0.3
}

SIMPLE2v1 = {
    "game": {
        "num_obsts": 0,
        "num_preds": 2,
        "num_preys": 1,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0, # 6.0
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 500
    }
}

SIMPLE2v2 = {
    "game": {
        "num_obsts": 1,
        "num_preds": 2,
        "num_preys": 2,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 500
    }
}