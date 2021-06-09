
OBSTACLES_STATE_DIM = 3
OBSTACLES_TEAM_SIZE = 1

PREY_STATE_DIM = 4
PREY_TEAM_SIZE = 2

PREDATOR_STATE_DIM = 3
PREDATOR_TEAM_SIZE = 2

TEAMS_STATE_SIZE = PREDATOR_STATE_DIM * PREDATOR_TEAM_SIZE + PREY_STATE_DIM * PREY_TEAM_SIZE
OBSTACLES_STATE_SIZE = OBSTACLES_TEAM_SIZE * OBSTACLES_STATE_DIM

CRITIC_ACTION_SIZE = PREY_TEAM_SIZE + PREDATOR_TEAM_SIZE

buffer_config = {
    "size": 250_000,
    "n_agents": 2,
    "state_size": TEAMS_STATE_SIZE + OBSTACLES_STATE_SIZE,
    "action_sizes": [PREDATOR_TEAM_SIZE, PREY_TEAM_SIZE]
}


prey_agent_config = {
    "teams_state_size": TEAMS_STATE_SIZE,
    "obstacle_state_size": OBSTACLES_STATE_SIZE,
    "actor_action_size": PREY_TEAM_SIZE,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3, 
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 1.0
}

predator_agent_config = {
    "teams_state_size": TEAMS_STATE_SIZE,
    "obstacle_state_size": OBSTACLES_STATE_SIZE,
    "actor_action_size": PREDATOR_TEAM_SIZE,
    "critic_action_size": CRITIC_ACTION_SIZE,
    "actor_hidden_size": 64, 
    "critic_hidden_size": 64, 
    "actor_lr": 1e-3, 
    "critic_lr": 1e-3, 
    "tau": 1e-3, 
    "gamma": 0.99, 
    "act_noise": 1.0
}

SIMPLE1v1 = {
    "game": {
        "num_obsts": 1,
        "num_preds": 1,
        "num_preys": 1,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 1.5],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 500
    }
}


SIMPLE2v1 = {
    "game": {
        "num_obsts": 4,
        "num_preds": 2,
        "num_preys": 1,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 1.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 500
    }
}


SIMPLE1v2 = {
    "game": {
        "num_obsts": 0,
        "num_preds": 1,
        "num_preys": 2,
        "x_limit": 4,
        "y_limit": 4,
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


SIMPLE2v2 = {
    "game": {
        "num_obsts": 1,
        "num_preds": 2,
        "num_preys": 2,
        "x_limit": 7,
        "y_limit": 7,
        "obstacle_radius_bounds": [0.8, 1.5], # [0.8, 2.0]
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 500
    }
}

SIMPLE2v5 = {
    "game": {
        "num_obsts": 0,
        "num_preds": 2,
        "num_preys": 5,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 1.5], # [0.8, 2.0]
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 500
    }
}


SUBMISSION2v5 = {
    "game": {
        "num_obsts": 10,
        "num_preds": 2,
        "num_preys": 5,
        "x_limit": 9,
        "y_limit": 9,
        "obstacle_radius_bounds": [0.8, 1.5],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 1000
    }
}