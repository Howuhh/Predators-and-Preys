from .physics.entity import Entity
import random


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

        self.random = random.Random()

    def seed(self, seed):
        self.random = random.Random(seed)

    def get_state_dict(self):
        state_dict = {
            "predators": [],
            "preys": [],
            "obstacles": []
        }

        for is_collide, e in self.predators:
            state_dict["predators"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius,
                "speed": e.speed
            })
        for is_alive, e in self.preys:
            state_dict["preys"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius,
                "speed": e.speed,
                "is_alive": is_alive
            })
        for e in self.obstacles:
            state_dict["obstacles"].append({
                "x_pos": e.position[0],
                "y_pos": e.position[1],
                "radius": e.radius
            })
        return state_dict
    
    def get_reward_dict(self):
        
        reward_dict = {
            "predators": [],
            "preys": []
        }
        
        for is_collide, e in self.predators:
            reward_dict["predators"].append({
                "reward": self._get_predator_reward(is_collide, e),
            })
        
        for is_alive, e in self.preys:
            reward_dict["preys"].append({
                "reward": self._get_prey_reward(is_alive, e),
            })
        
        return reward_dict

    def _get_predator_reward(self, is_collide, agent):
        reward = 0.0
        
        if is_collide:
            for is_alive, e in self.preys:
                # if not is_alive:
                if agent.is_intersect(e):
                    reward += 10
                    
        # reward -= 0.1 * min([agent.real_distance(e) for is_alive, e in self.preys if is_alive], default=0.0) 
        
        return reward
    
    def _get_prey_reward(self, is_alive, agent):
        reward = 0.0
        
        if not is_alive:
            for is_collide, e in self.predators:
                # if is_collide:
                if agent.is_intersect(e):
                    reward -= 10
        
        # reward += 0.1 * min([agent.real_distance(e) for _, e in self.predators])
        
        return reward

    def step(self, actions):
        assert len(actions["preys"]) == len(self.preys)
        assert len(actions["predators"]) == len(self.predators)
        
        for a, (is_alive, e) in zip(actions["preys"], self.preys):
            if not is_alive:
                continue
            e.move(a, self.world_timestep)
        for a, (is_collide, e) in zip(actions["predators"], self.predators):
            e.move(a, self.world_timestep)

        corrected = True
        while corrected:
            corrected = False
            for i, (is_alive, e) in enumerate(self.preys):
                this_corrected = False
                e.force_clip_positon(-self.x_limit, -self.y_limit, self.x_limit, self.y_limit)
                for other in self.obstacles:
                    this_corrected = this_corrected or e.force_not_intersect(other)
                if not this_corrected:
                    for j, (_, other) in enumerate(self.preys):
                        if i == j:
                            continue
                        this_corrected = this_corrected or e.force_not_intersect(other)
                if not this_corrected:
                    for other in self.obstacles:
                        this_corrected = this_corrected or e.force_not_intersect(other)
                corrected = corrected or this_corrected
                e.force_clip_positon(-self.x_limit, -self.y_limit, self.x_limit, self.y_limit)
            if not corrected:
                break

        corrected = True
        while corrected:
            corrected = False
            for i, (is_collide, e) in enumerate(self.predators):
                this_corrected = False
                e.force_clip_positon(-self.x_limit, -self.y_limit, self.x_limit, self.y_limit)
                for other in self.obstacles:
                    this_corrected = this_corrected or e.force_not_intersect(other)
                if not this_corrected:
                    for j, (_, other) in enumerate(self.predators):
                        if i == j:
                            continue
                        this_corrected = this_corrected or e.force_not_intersect(other)
                if not this_corrected:
                    for other in self.obstacles:
                        this_corrected = this_corrected or e.force_not_intersect(other)
                corrected = corrected or this_corrected
                e.force_clip_positon(-self.x_limit, -self.y_limit, self.x_limit, self.y_limit)
            if not corrected:
                break

        for i in range(len(self.preys)):
            for j in range(len(self.predators)):
                if self.predators[j][1].is_intersect(self.preys[i][1]):
                    self.preys[i][0] = False
                    self.predators[j][0] = True

    def reset(self):
        self.obstacles = []
        for _ in range(self.num_obsts):
            r = self.random.random() * (self.obstacle_r_b[1] - self.obstacle_r_b[0]) + self.obstacle_r_b[0]
            x = (2 * self.random.random() - 1) * (self.x_limit - r)
            y = (2 * self.random.random() - 1) * (self.y_limit - r)
            self.obstacles.append(Entity(r, 0., x, y))

        self.preys = []
        for _ in range(self.num_preys):
            created = False
            while not created:
                created = True
                x = (2 * self.random.random() - 1) * (self.x_limit - self.prey_radius)
                y = (2 * self.random.random() - 1) * (self.y_limit - self.prey_radius)
                new_e = Entity(self.prey_radius, self.prey_speed, x, y)
                for e in self.obstacles:
                    if e.is_intersect(new_e):
                        created = False
                        break

                if created:
                    for _, e in self.preys:
                        if e.is_intersect(new_e):
                            created = False
                            break

                if created:
                    self.preys.append([True, new_e])

        self.predators = []
        for _ in range(self.num_preds):
            created = False
            while not created:
                created = True
                x = (2 * self.random.random() - 1) * (self.x_limit - self.predator_radius)
                y = (2 * self.random.random() - 1) * (self.y_limit - self.predator_radius)
                new_e = Entity(self.predator_radius, self.predator_speed, x, y)
                for e in self.obstacles:
                    if e.is_intersect(new_e):
                        created = False
                        break

                if created:
                    for _, e in self.preys:
                        if e.is_intersect(new_e):
                            created = False
                            break
                if created:
                    for _, e in self.predators:
                        if e.is_intersect(new_e):
                            created = False
                            break

                if created:
                    self.predators.append([False, new_e])
