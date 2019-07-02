from src.utils import random, array
import numpy as np
from .kd_tree import KDTree

# Interface with controller
# GameState calls world interface
class GameState:
    """
            A class used to communicate with the model

            ...

            Attributes
            ----------
            player_list : list
                a list of players id that are currently in the game
            world: list
                a list of all game objects and their positions

            Methods
            -------
            get_objects_in_region(top_left, bottom_right):
                Return positions of objects in specific area

            update()
                Return states and positions of all objects a each time iteration

            create_ants(position, amount)
                Return new ant objects

    """

    def __init__(self, player_list):
        """ Initialize player list and create nests for all the players

        :param player_list: (list) that contains current players IDs

        """
        self.players = player_list
        self.world = KDTree()
        positions = []
        for i in range(len(player_list)):
            positions.append(random(2) * 250)
        self.world.create_nests(player_list, positions, health=100, size=10)
        self.generate_random_food(array([-250, 250]), array([250, -250]), 50, [5] * 50)

    def get_objects_in_region(self, top_left, bottom_right):
        """ Get list of positions and all included objects (ants, nests, foods, pheromones, etc) in a specific
            rectangular area

        :param top_left: (ndarray) Coordinates of top left point of the rectangle
        :param bottom_right: (ndarray) Coordinates of bottom right point of the rectangle
        :return:

        """
        return self.world.get_rectangle_region(top_left, bottom_right)

    def update(self):
        """Return the states of all the objects and their positions at each time iteration """
        self.world.update()

    def create_ants(self, nest, ant_type="worker", amount=1):
        """Create new ant objects in the specific nest with the given positions

        :param nest: nest
        :param ant_type: (string) Has to be one of "worker" or "scout", defines the type of Ant to be created
        :param amount: (int) number of ants that should be created
        :return:

        """
        return self.world.create_ants(nest, ant_type, amount)

    def create_nest(self, nest_position, player, size, health):
        return self.world.create_nests(nest_position, player, size, health)

    def create_food(self, position_list, size_list):
        return self.world.create_food(position_list, size_list)

    def get_nests(self):
        return self.world.get_nests()

    def get_ants(self):
        return self.world.get_ants()

    def generate_random_food(self, top_left, bottom_right, amount, size_list):
        position_list = []
        for i in range(amount):
            x_span = bottom_right[0] - top_left[0]
            x_position = top_left[0] + x_span * random(1)
            y_span = top_left[1] - bottom_right[1]
            y_position = top_left[0] + y_span * random(1)
            position_list.append(np.concatenate((x_position, y_position)))
        self.create_food(position_list, size_list)
