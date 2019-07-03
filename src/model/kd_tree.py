from scipy.spatial import cKDTree
import numpy as np

from .worker import Worker
from .ant import Ant
from .scout import Scout
from .food import Food
from .nest import Nest

from src.utils import empty
from src.settings import all_params
from .world import World

class KDTree(World):
    def __init__(self):
        self.all_objects = []
        self.tree = None

    def get_k_nearest(self, position, k=1):
        """ Get k nearest neighbour ants for specific position using kd_tree that uses Euclidean distance.
        :param position: (list) Coordinates of the position of interest
        :param k: (int) Number of nearest neighbours
        :return point_matrix: (array) ### Add this
        :return dists: (array of floats) Distances to the nearest neighbours
        """
        dists, idx = self.tree.query(position, k, p=2, n_jobs=-1)

        k_nearest_obj = []
        for i in idx:
            k_nearest_obj.append(self.all_object[i])

        return dists, k_nearest_obj

    def get_at_position(self, position):
        """ Return all the objects (ants/food/nest) in specific position
        :param position: (list) Coordinates of specific position
        :return: (list) ALL objects in the given position
        """
        return_idx = self.tree.query_ball_point(position, r=0)
        obj_at_pos = []
        for obj in return_idx:
            obj_at_pos.append(self.all_objects[obj])

        return obj_at_pos

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
        position_idx = self.tree.query_ball_point(center, radius, p=2, n_jobs=-1)
        obj_at_pos = []
        for i in position_idx:
            obj_at_pos.append(self.all_objects[i])

        return obj_at_pos

    # def get_k_nearest_list(self, position_list, k):
    #     """ Get k nearest neighbour ants for list of positions using kd_tree that uses Euclidean distance.
    #
    #     :param position_list: (list) Coordinates of the positions of interests
    #     :param k: (int) Number of nearest neighbours
    #     :return result: (list) all nearest neighbours objects
    #     :return dists: (array of floats) Distances to the nearest neighbours
    #
    #     """
    #
    #     if len(position_list) == 1:
    #         return self.get_k_nearest(position_list, k)
    #     dists, idx_list = self.kd_tree.query(position_list, k, p=2, n_jobs=-1)
    #     result = []
    #     for idx in idx_list:
    #         if len(idx.shape) == 0:
    #             result.append(self.all_objects[idx])
    #         else:
    #             sub_result = [obj for obj in self.all_objects[idx]]
    #             dict_ind = self.point_matrix[idx]
    #             sub_result = [obj for row in dict_ind for obj in self.all_objects[tuple(row)]]
    #             result.append(sub_result)
    #
    #     return result, dists

    # def get_circular_region_list(self, center_list, radius_list):
    #     """ Return all the objects in each of the circles of interest
    #
    #     :param center_list: (list) Coordinates of the centers of circles of interest
    #     :param radius_list: (list) Radii of the circles of interest
    #     :return: (list) All objects in each of the specified circular region
    #
    #     """
    #
    #     position_idx_list = self.kd_tree.query_ball_point(center_list, radius_list,
    #                                                       p=2, n_jobs=-1)
    #     result = []
    #     for position_idx in position_idx_list:
    #         positions = self.point_matrix[position_idx]
    #         sub_result = []
    #         for position in positions:
    #             sub_result.extend(self.all_objects.get(tuple(position)))
    #         result.append(sub_result)
    #     return result

    def update(self):
        all_obj_new = []

        for obj in self.all_objects:
            if isinstance(obj, Ant):
                if isinstance(obj, Scout):
                    noticeable_objects = self.get_circular_region(
                        obj.position, radius=all_params.tree_model_params.circular_region_radius_scout)

                elif isinstance(obj, Worker):
                    noticeable_objects = self.get_circular_region(
                        obj.position, radius=all_params.tree_model_params.circular_region_radius_worker)

                new_position, new_pheromone = obj.update(noticeable_objects)

                if new_pheromone is not None:
                    all_obj_new.append(new_pheromone)
            else:
                new_position = obj.update()

            if new_position is not None:
                all_obj_new.append(obj)

        self.all_objects = all_obj_new
        self._update_tree()

    def create_nests(self, player_list, position_list, size, health):
        """ Create new nest objects with specific owners/positions/size/health and update the tree
        :param player_list: (list) owning players of the nests to be created
        :param position_list: (list) coordinates of the nests to be created
        :param size: (list) sizes of the nests to be created
        :param health: (list) health(s) of the nests to be created
        """

        for position, player in zip(position_list, player_list):
            self.all_objects.append(Nest(position, player, size, health))

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

        for _ in range(amount):
            self.all_objects.append(CorrectAnt(nest.owner, nest))

        self._update_tree()

    def create_food(self, position_list, size_list):
        """ Create new food objects with specific positions/size and update the tree
        :param position_list: (list) coordinates of the food to be created
        :param size_list: (list) size of the food to be created
        """

        # TODO: compare to extend with food list
        for position, size in zip(position_list, size_list):
            self.all_objects.append(Food(position, size))

        self._update_tree()

    def __iter__(self):
        return iter(self.all_objects)

    def __len__(self):
        return len(self.all_objects)
    
    def dump_content(self):
        return self.all_objects
    
    def _update_tree(self):
        """Update the tree"""
        all_pos = empty((len(self.all_objects), 2))
        for i, o in enumerate(self.all_objects):
            all_pos[i, :] = o.position
        self.tree = cKDTree(all_pos)

    def get_square_region(self, center, radius):
        """ Return all the objects in the given square region
        :param center: (list) Coordinates of center of the square
        :param radius: (int) Radius of the square
        :return result: (list) All objects in the specified circular region
        """

        position_idx = self.tree.query_ball_point(center, radius, p=np.inf, n_jobs=-1)
        obj_in_square = []
        for i in position_idx:
            obj_in_square.append(self.all_objects[i])

        return obj_in_square

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
        return self._get_all_from_type(Ant)

    def get_nests(self):
        """ Get all the nest objects
        :return: (list) all the nest objects
        """
        return self._get_all_from_type(Nest)

    def _get_all_from_type(self, type):
        all_wanted_obj = [obj for obj in self.all_objects if isinstance(obj, type)]
        return all_wanted_obj

