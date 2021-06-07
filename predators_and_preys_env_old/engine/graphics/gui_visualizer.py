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
        to_break = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    to_break = True
                if event.key == pygame.K_q:
                    quit()

        sd = self.game.get_state_dict()
        self.screen.fill((255, 255, 255))
        for e in sd["obstacles"]:
            pygame.draw.circle(self.screen, (160, 160, 160),
                               ((e["x_pos"] + self.x_bias) * self.ratio,
                                (e["y_pos"] + self.y_bias) * self.ratio),
                                 e["radius"] * self.ratio)

        for e in sd["predators"]:
            pygame.draw.circle(self.screen, (220, 0, 0),
                               ((e["x_pos"] + self.x_bias) * self.ratio,
                                (e["y_pos"] + self.y_bias) * self.ratio),
                                 e["radius"] * self.ratio)

        for e in sd["preys"]:
            is_alive = e["is_alive"]
            if is_alive:
                pygame.draw.circle(self.screen, (0, 220, 0),
                                   ((e["x_pos"] + self.x_bias) * self.ratio,
                                    (e["y_pos"] + self.y_bias) * self.ratio),
                                     e["radius"] * self.ratio)
            else:
                pygame.draw.circle(self.screen, (70, 70, 70),
                                   ((e["x_pos"] + self.x_bias) * self.ratio,
                                    (e["y_pos"] + self.y_bias) * self.ratio),
                                     e["radius"] * self.ratio)

        pygame.display.update()
        
        return to_break

