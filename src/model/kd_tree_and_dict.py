from scipy.spatial import cKDTree
import numpy as np

from .worker import Worker
from .scout import Scout
from .ant import Ant
from .food import Food
from .nest import Nest
from .world import World

from src.utils import array
from src.settings import all_params


class KdTreeAndDict(World):
    """
            A class used to implement the tree and dictionary in the game (alternative to gird)
            It inherits from World class

            ...

            Attributes
            ----------
            all_objects: dict
                a dictionary for all the objects in the game
            kd_tree: cKDTree
                a tcKDTree implementation for nearest neighbours
            point_matrix: array
                an ### Add this

    """

    def __init__(self):
        self.all_objects = {}
        self.kd_tree = None
        self.point_matrix = None

    def get_k_nearest(self, position, k=1):
        """ Get k nearest neighbour ants for specific position using kd_tree that uses Euclidean distance.

        :param position: (list) Coordinates of the position of interest
        :param k: (int) Number of nearest neighbours
        :return point_matrix: (array) ### Add this
        :return dists: (array of floats) Distances to the nearest neighbours

        """
        dists, idx = self.kd_tree.query(position, k, p=2, n_jobs=-1)
        dict_idxs = self.point_matrix[idx]
        if k == 1:
            game_object_list = self.all_objects.get(tuple(dict_idxs), [])
        else:
            game_object_list = [obj for idx in dict_idxs for obj in self.all_objects.get(tuple(idx))]
        return game_object_list, dists

    def get_at_position(self, position):
        """ Return all the objects (ants/food/nest) in specific position

        :param position: (list) Coordinates of specific position
        :return: (list) ALL objects in the given position

        """
        return self.all_objects.get(tuple(position))

    def get_rectangle_region(self, top_left, bottom_right):
        """ Return all the objects in the given rectangular region

        :param top_left: (list) Coordinates of top left point of the rectangle
        :param bottom_right: (list) Coordinates of bottom right point of the rectangle
        :return result: (list) All objects in the specified rectangular region

        """
        longest_side = max(bottom_right[0] - top_left[0], top_left[1] - bottom_right[1])
        center = (top_left + bottom_right) / 2
        large_square_objects = self.get_square_region(center, longest_side)

        x_min = top_left[0]
        y_min = bottom_right[1]
        x_max = bottom_right[0]
        y_max = top_left[1]

        result = []
        for obj in large_square_objects:
            if self._is_in_rectangle(obj.position, x_min, x_max, y_min, y_max):
                result.append(obj)
        return result

    def get_circular_region(self, center, radius):
        """ Return all the objects in the given circular region

        :param center: (list) Coordinates of center of the circle
        :param radius: (int) Radius of the circle
        :return result: (list) All objects in the specified circular region

        """
        position_idx = self.kd_tree.query_ball_point(center, radius, p=2, n_jobs=-1)
        positions = self.point_matrix[position_idx]
        result = []
        for position in positions:
            result.extend(self.all_objects.get(tuple(position)))
        return result

    def get_k_nearest_list(self, position_list, k):
        """ Get k nearest neighbour ants for list of positions using kd_tree that uses Euclidean distance.
    
        :param position_list: (list) Coordinates of the positions of interests
        :param k: (int) Number of nearest neighbours
        :return result: (list) all nearest neighbours objects
        :return dists: (array of floats) Distances to the nearest neighbours
    
        """
    
        if len(position_list) == 1:
            return self.get_k_nearest(position_list, k)
        dists, idx_list = self.kd_tree.query(position_list, k, p=2, n_jobs=-1)
        result = []
        for idx in idx_list:
            if len(idx.shape) == 0:
                dict_ind = self.point_matrix[idx]
                result.append(self.all_objects[tuple(dict_ind)])
            else:
                dict_ind = self.point_matrix[idx]
                sub_result = [obj for row in dict_ind for obj in self.all_objects[tuple(row)]]
                result.append(sub_result)
    
        return result, dists
    
    def get_circular_region_list(self, center_list, radius_list):
        """ Return all the objects in each of the circles of interest
    
        :param center_list: (list) Coordinates of the centers of circles of interest
        :param radius_list: (list) Radii of the circles of interest
        :return: (list) All objects in each of the specified circular region
    
        """
    
        position_idx_list = self.kd_tree.query_ball_point(center_list, radius_list,
                                                          p=2, n_jobs=-1)
        result = []
        for position_idx in position_idx_list:
            positions = self.point_matrix[position_idx]
            sub_result = []
            for position in positions:
                sub_result.extend(self.all_objects.get(tuple(position)))
            result.append(sub_result)
        return result

    # @timing # TODO: check in real game how much time is needed
    def update(self):
        """ Update the positions of all ants after their movement in one iteration and remove the previous positions"""
        all_lists = list(self.all_objects.values())
        for listy in all_lists:
            for item in listy:
                old_position = tuple(item.position)

                if isinstance(item, Ant):
                    # TODO: pick radius (or implement it in ant class)
                    if isinstance(item, Scout):
                        radius = all_params.tree_model_params.circular_region_radius_scout
                    elif isinstance(item, Worker):
                        radius = all_params.tree_model_params.circular_region_radius_worker

                    noticeable_objects = self.get_circular_region(item.position, radius=radius)
                    new_position, new_pheromone = item.update(noticeable_objects)

                    # Only handle if new pheromone object needs to be created.
                    if new_pheromone is not None:
                        pheromone_pos = tuple(new_pheromone.position)
                        self.all_objects.setdefault(pheromone_pos, []).append(new_pheromone)

                else:
                    new_position = item.update()

                # Remove old positions.
                self.all_objects[old_position].remove(item)

                if self.all_objects[old_position] is []:
                    self.all_objects.pop(old_position)

                # Save new positions if object did not die.
                # TODO: deaths of objects
                if new_position is not None:
                    self.all_objects.setdefault(tuple(new_position), []).append(item)
        self._update_tree()

    def create_nests(self, player_list, position_list, size, health):
        """ Create new nest objects with specific owners/positions/size/health and update the tree

        :param player_list: (list) owning players of the nests to be created
        :param position_list: (list) coordinates of the nests to be created
        :param size: (list) sizes of the nests to be created
        :param health: (list) health(s) of the nests to be created

        """

        for position, player in zip(position_list, player_list):
            self.all_objects.setdefault(tuple(position), []).append(Nest(position, player, size, health))
        self._update_tree()

    def create_ants(self, nest, ant_type, amount):
        """ Create new ant objects in a specific nest with the given amount and update the tree

        :param nest: nest object where new ants should be created
        :param amount: (int) number of ants that should be created

        """

        if ant_type == "worker":
            CorrectAnt = Worker
        elif ant_type == "scout":
            CorrectAnt = Scout
        else:
            raise ValueError("Incorrect Ant type passed at ant creation.")

        player = nest.owner
        position = nest.position
        for _ in range(amount):
            self.all_objects.setdefault(tuple(position), []).append(CorrectAnt(player, nest))
        self._update_tree()

    def create_food(self, position_list, size_list):
        """ Create new food objects with specific positions/size and update the tree

        :param position_list: (list) coordinates of the food to be created
        :param size_list: (list) size of the food to be created

        """

        # TODO: compare to extend with food list
        for position, size in zip(position_list, size_list):
            self.all_objects.setdefault(tuple(position), []).append(Food(position, size))
        self._update_tree()

    def dump_content(self):
        everything = list(self.all_objects.values())
        flat_everything = [obj for listy in everything for obj in listy]
        return flat_everything

    def __iter__(self):
        """
        For iterating over the tree.
        :return: iterator of all objects currently saved in tree
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        return iter(flatten(list(self.all_objects.values())))

    def __len__(self):
        """
        :return: number of objects that are in the tree
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        return len(flatten(list(self.all_objects.values())))

    def _update_tree(self):
        """Update the tree"""
        keys = list(self.all_objects.keys())
        self.point_matrix = array(keys)
        self.kd_tree = cKDTree(self.point_matrix)

    def get_square_region(self, center, radius):
        """ Return all the objects in the given square region

        :param center: (list) Coordinates of center of the square
        :param radius: (int) Radius of the square
        :return result: (list) All objects in the specified circular region

        """

        position_idx = self.kd_tree.query_ball_point(center, radius, p=np.inf, n_jobs=-1)
        positions = self.point_matrix[position_idx]
        result = []
        for position in positions:
            result.extend(self.all_objects.get(tuple(position)))
        return result

    def _is_in_rectangle(self, position, x_min, x_max, y_min, y_max):
        """ Decide whether specific object is in the rectangular area of interest

        :param position: (list) coordinates of object
        :param x_min: (int) minimum x coordinate in the rectangle of interest
        :param x_max: (int) maximum x coordinate in the rectangle of interest
        :param y_min: (int) minimum y coordinate in the rectangle of interest
        :param y_max: (int) maximum y coordinate in the rectangle of interest
        :return: (boolean) TRUE if an ibject is indeed in the specified rectangular area.

        """

        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def get_ants(self):
        """ Get all the ant objects

        :return: (list) all the ant objects

        """

        everything = self.dump_content()
        ants = [obj for obj in everything if isinstance(obj, Ant)]
        return ants

    def get_nests(self):
        """ Get all the nest objects

        :return: (list) all the nest objects
        """
        everything = self.dump_content()
        nests = [obj for obj in everything if type(obj) is Nest]
        return nests
