# Interface used by GameState
# Implementation is done currently by KdTreeAndDict
from abc import ABC, abstractmethod


class World(ABC):
    """

            Abstract class represents interface used by GameState.
            It inherits from ABC.
            All the abstract methods here will be overridden by their implementation in kd_tree_and_dict.py

    """

    @abstractmethod
    def get_k_nearest(self, position, k):
        """ Return k nearest ants to specific position
            The implementation depends on the subclass.

        """
        # Inheriting class should return k nearest neighbours of the passed position
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def get_at_position(self, position):
        """ Return all the objects (ants/food/nest) in specific given position"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def get_rectangle_region(self, top_left, bottom_right):
        """Return all the objects in the given rectangular region"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def get_circular_region(self, center, radius):
        """Return all the objects in the given circular region"""
        raise NotImplementedError("Please use subclassing.")

    # @abstractmethod
    # def get_k_nearest_list(self, position_list, k):
    #     """ Get k nearest neighbour ants for list of positions using kd_tree that uses Euclidean distance."""
    #     # Inheriting class should return k nearest neighbours of the passed position
    #     raise NotImplementedError("Please use subclassing.")

    # @abstractmethod
    # def get_circular_region_list(self, center_list, radius_list):
    #     """ Return all the objects in each of the circles of interest"""
    #     raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def update(self):
        """ Update the positions of all ants after their movement in one iteration and remove the previous positions"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def create_nests(self, color_list, position_list, size, health):
        """Create new nests with specific colors/positions/size/health and update the tree"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def create_ants(self, nest, ant_type, amount):
        """ Create new ant objects in a specific nest with the given amount and update the tree"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def create_food(self, position_list, size_list):
        """Create new food objects with specific positions/size and update the tree"""
        raise NotImplementedError("Please use subclassing.")

    # @abstractmethod
    # def dump_content(self):
    #     raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def get_ants(self):
        """Get all the ant objects"""
        raise NotImplementedError("Please use subclassing.")

    @abstractmethod
    def get_nests(self):
        """Get all the nest objects"""
        raise NotImplementedError("Please use subclassing.")

