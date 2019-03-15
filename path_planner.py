from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node

        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse

        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        found = False
        pq = []
        node = start_position
        cost = self.node_grid.get_node(node[0], node[1]).f = 0
        heapq.heappush(pq, (cost, node))

        while len(pq) != 0 and not found:

            (cost, node) = heapq.heappop(pq)
            node_node = self.node_grid.get_node(node[0], node[1])
            node_node.closed = True

            for successor in self.node_grid.get_successors(node[0], node[1]):
                successor_cost = self.node_grid.get_node(successor[0], successor[1]).f
                calculated_cost = cost + self.cost_map.get_edge_cost(node, successor)
                if successor_cost > calculated_cost:
                    successor_node = self.node_grid.get_node(successor[0], successor[1])
                    successor_cost = successor_node.f = calculated_cost
                    heapq.heappush(pq, (successor_cost, successor))
                    successor_node.parent = node_node
                if successor == goal_position:
                    found = True
                    break

        path = PathPlanner.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
        cost = self.node_grid.get_node(goal_position[0], goal_position[1]).f
        self.node_grid.reset()

        return path, cost

    @staticmethod
    def heuristic(node, goal_position):
        return ((node[0] - goal_position[0])**2 + (node[1] - goal_position[1])**2)**(1/2)

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.
        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        found = False
        pq = []
        node = start_position
        cost = PathPlanner.heuristic(node, goal_position)
        self.node_grid.get_node(node[0], node[1]).f = 0
        heapq.heappush(pq, (cost, node))

        while len(pq) != 0 and not found:
            (cost, node) = heapq.heappop(pq)
            node_node = self.node_grid.get_node(node[0], node[1])
            node_node.closed = True
            for successor in self.node_grid.get_successors(node[0], node[1]):
                successor_node = self.node_grid.get_node(successor[0], successor[1])
                if not successor_node.closed:
                    calculated_cost = node_node.f + self.cost_map.get_edge_cost(node, successor)
                    successor_cost = PathPlanner.heuristic(successor, goal_position)
                    successor_node.parent = node_node
                    successor_node.f = calculated_cost
                    heapq.heappush(pq, (successor_cost, successor))
                    if successor == goal_position:
                        found = True
                        break

        path = PathPlanner.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
        cost = self.node_grid.get_node(goal_position[0], goal_position[1]).f

        self.node_grid.reset()

        return path, cost

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        # found = False
        pq = []
        node = start_position
        node_node = self.node_grid.get_node(node[0], node[1])

        node_node.g = 0
        node_node.f = PathPlanner.heuristic(node, goal_position)

        heapq.heappush(pq, (node_node.f, node))

        while len(pq) != 0:
            (cost, node) = heapq.heappop(pq)
            node_node = self.node_grid.get_node(node[0], node[1])
            node_node.closed = True

            if node == goal_position:
                # print('aaaa')
                break
            # print(1)
            for successor in self.node_grid.get_successors(node[0], node[1]):
                successor_node = self.node_grid.get_node(successor[0], successor[1])

                g_calculated_cost = node_node.g + self.cost_map.get_edge_cost(node, successor)
                f_calculated_cost = g_calculated_cost + PathPlanner.heuristic(successor, goal_position)

                if successor_node.f > f_calculated_cost:
                    # if not successor_node.closed:
                    successor_node.g = g_calculated_cost
                    successor_node.f = f_calculated_cost
                    successor_node.parent = node_node
                    heapq.heappush(pq, (successor_node.f, successor))

        path = PathPlanner.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
        cost = self.node_grid.get_node(goal_position[0], goal_position[1]).g

        self.node_grid.reset()
        return path, cost
