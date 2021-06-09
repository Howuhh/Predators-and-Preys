from ctypes import *
import pathlib
import numpy as np
import platform
import os

suff = ".so" if platform.system() == "Linux" else ".dylib"  # No Windows

dir_path = os.path.dirname(__file__)

lib_g = CDLL(dir_path + "/game" + suff)
lib_e = CDLL(dir_path + "/physics/entity" + suff)

class Entity(Structure):
    _fields_ = [('radius', c_double), ('speed', c_double), ('position', c_double * 2)]

entity_p = POINTER(Entity)
int_p = POINTER(c_int)
double_p = POINTER(c_double)

class FGame(Structure):
    _fields_ = [("predators", entity_p),
                ("preys", entity_p),
                ("alive", int_p),
                ("obstacles", entity_p),
                ("x_limit", c_double),
                ("y_limit", c_double),
                ("num_preds", c_int),
                ("num_preys", c_int),
                ("num_obstacles", c_int),
                ("r_obst_ub", c_double),
                ("r_obst_lb", c_double),
                ("prey_radius", c_double),
                ("pred_radius", c_double),
                ("pred_speed", c_double),
                ("prey_speed", c_double),
                ("w_timestep", c_double),
                ("frameskip", c_int),
                ("frame_count", c_int),
                ("prey_order", int_p),
                ("pred_order", int_p),
                ("preys_reward", double_p),
                ("preds_reward", double_p),
                ("prey_mask", int_p),
                ("pred_mask", int_p),
                ("max_dist", c_double),
                ("min_dist", c_double),
                ("al", c_int)]
                
game_p = POINTER(FGame)


ginit = lib_g.game_init
ginit.argtypes = [c_double, c_double, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_double, c_double]
ginit.restype = game_p


get_predator = lib_g.get_predator
get_predator.argtypes = [game_p, c_int]
get_predator.restype = Entity


get_prey = lib_g.get_prey
get_prey.argtypes = [game_p, c_int]
get_prey.restype = Entity

get_alive = lib_g.get_alive
get_alive.argtypes = [game_p, c_int]
get_alive.restype = c_int


get_obstacle = lib_g.get_obstacle
get_obstacle.argtypes = [game_p, c_int]
get_obstacle.restype = Entity


Step = lib_g.step
Step.argtypes = [game_p, double_p, double_p]

Reset = lib_g.reset
Reset.argtypes = [game_p]

seeding = lib_g.seed
seeding.argtypes = [c_int]

class Game:
    def __init__(self, config):
        self.config = config
        self.predators = []
        self.preys = []
        self.obstacles = [] 
        self.x_limit = config["x_limit"]      # 16
        self.y_limit = config["y_limit"]      # 16
        self.num_preds = config["num_preds"]  # 3
        self.num_preys = config["num_preys"]  # 5
        self.num_obsts = config["num_obsts"]  # 10

        self.obstacle_r_b = config["obstacle_radius_bounds"]  # 0.6 .. 1.2
        self.prey_radius = config["prey_radius"]              # 0.8
        self.predator_radius = config["predator_radius"]      # 1.0
        self.predator_speed = config["predator_speed"]        # 1.0
        self.prey_speed = config["prey_speed"]                # 1.2
        self.world_timestep = config["world_timestep"]        # 1/60
        self.frameskip = config["frameskip"]

        self.true_game = ginit(self.x_limit, self.y_limit,
                               self.num_preds, self.num_preys, self.num_obsts,
                               self.obstacle_r_b[0], self.obstacle_r_b[1],
                               self.prey_radius, self.predator_radius,
                               self.predator_speed, self.prey_speed,
                               self.world_timestep, self.frameskip)
                          
    def seed(self, n):
        seeding(n)
        
    def get_state_dict(self):
        state_dict = { 
            "predators": [],
            "preys": [],
            "obstacles": []
        }
        for i in range(self.num_preds):
            e = get_predator(self.true_game, i)
            state_dict["predators"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius,
                "speed": e.speed
            })
        for i in range(self.num_preys):
            e = get_prey(self.true_game, i)
            ia = get_alive(self.true_game, i)
            state_dict["preys"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius,
                "speed": e.speed,
                "is_alive": ia
            })
        for i in range(self.num_obsts):
            e = get_obstacle(self.true_game, i)
            state_dict["obstacles"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius
            })
        return state_dict
    
    def get_reward(self):
        reward = {}
        reward["preys"] = np.ctypeslib.as_array(self.true_game.contents.preys_reward, shape=(self.num_preys,))
        reward["predators"] = np.ctypeslib.as_array(self.true_game.contents.preds_reward, shape=(self.num_preds,))
        return reward
    
    def step(self, actions):
        Step(self.true_game, 
             np.array(actions["preys"], dtype=np.float64).ctypes.data_as(double_p),
             np.array(actions["predators"], dtype=np.float64).ctypes.data_as(double_p))
        
    def reset(self):
        Reset(self.true_game)
