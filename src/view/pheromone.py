import pygame
from .view_element import ViewElement

class Pheromone(ViewElement):
    def __init__(self, view, identifier, x, y, value, color, max_value):
        super(Pheromone, self).__init__(view, identifier, x, y, width=4, height=4)
        self.z_index = 8
        self.value = value
        self.color = color
        self.max_value = max_value

    def draw(self):
        pheromone = pygame.Surface((self.width, self.height))
        pheromone.fill((255, 255, 255))
        pheromone.set_colorkey((255, 255, 255))
        
        pygame.draw.circle(pheromone, self.color, (int(self.width / 2), int(self.height / 2)), int(self.width / 2))
        pheromone.set_alpha(min(100, (self.value / self.max_value) * 100 + 20))
        
        self.view.screen.blit(pheromone, (self.x - self.width / 2, self.y - self.height / 2))
        
