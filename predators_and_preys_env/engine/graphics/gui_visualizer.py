import pygame


class GuiVisualizer:
    def __init__(self, game):
        pygame.init()

        x_size = game.x_limit
        y_size = game.y_limit
        ratio = y_size / x_size
        if ratio <= 1.:
            x_size = 640
            y_size = x_size * ratio
        else:
            y_size = 640
            x_size = y_size / ratio

        self.screen = pygame.display.set_mode((int(x_size), int(y_size)))
        self.game = game
        self.ratio = x_size / (game.x_limit * 2)
        self.x_bias = game.x_limit
        self.y_bias = game.y_limit

    def update(self):
        self.screen.fill((255, 255, 255))
        for entity in self.game.obstacles:
            pygame.draw.circle(self.screen, (160, 160, 160),
                               ((entity.position[0] + self.x_bias) * self.ratio,
                                (entity.position[1] + self.y_bias) * self.ratio),
                               entity.radius * self.ratio)

        for is_collide, entity in self.game.predators:
            pygame.draw.circle(self.screen, (220, 0, 0),
                               ((entity.position[0] + self.x_bias) * self.ratio,
                                (entity.position[1] + self.y_bias) * self.ratio),
                               entity.radius * self.ratio)

        for is_alive, entity in self.game.preys:
            if is_alive:
                pygame.draw.circle(self.screen, (0, 220, 0),
                                   ((entity.position[0] + self.x_bias) * self.ratio,
                                    (entity.position[1] + self.y_bias) * self.ratio),
                                   entity.radius * self.ratio)
            else:
                pygame.draw.circle(self.screen, (70, 70, 70),
                                   ((entity.position[0] + self.x_bias) * self.ratio,
                                    (entity.position[1] + self.y_bias) * self.ratio),
                                   entity.radius * self.ratio)

        pygame.display.update()

