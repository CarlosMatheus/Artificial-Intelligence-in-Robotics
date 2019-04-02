# Path Finder

Find a path from a source point and a destination using or not heuristics. Three methods were studied Dijkstra, A *, Grizzly Search. Of these only Dijkstra does not use heuristics.

## Run the code

Make sure to have python3 installed and pip.

Setup the environment:
```bash
cd Artificial-Intelligence-in-Robotics
python3 -m venv venv
source venv/bin/activate
pip install requirements.txt
```

Run:
```bash
cd Artificial-Intelligence-in-Robotics/path_planner/code
python main.py
```

## Some results

### Dijkstra Algorithm

The Dijkstra algorithm is an algorithm for determining the shortest path, given a vertex origin, to the rest of the vertices in a graph that has weights in each edge.

The underlying idea in this algorithm is to go exploring all the shortest paths that start from the vertex origin and that lead to all the other vertices; when the shortest path is obtained from the source vertex to the rest of the vertices that make up the graph, the algorithm stops. It is a specialization of the search for uniform cost and, as such, does not work in graphs with edges of negative cost (by always choosing the node with the least distance, they can be excluded from the search nodes that in future iterations would lower the general cost of the road when going through an edge with negative cost).

![Alt text](report/imgs/1.png?raw=true "Title")

Code idea:
```python
def dijkstra(self, start_position, goal_position):
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
```

### A*

The A* algorithm ("A star") belongs to the class of informed search algorithms. It is used in computer science to calculate a shortest path between two nodes in a graph with positive edge weights. It was first described in 1968 by Peter Hart, Nils J. Nilsson and Bertram Raphael. The algorithm is considered to be a generalization and extension of the Dijkstra algorithm, but in many cases A * can also be reduced to Dijkstra.

In contrast to uninformed search algorithms, the A* algorithm uses an estimator (heuristic) to search in a targeted way to reduce runtime. The algorithm is complete and optimal. This means that an optimal solution is always found, if one exists.

![Alt text](report/imgs/2.png?raw=true "Title")

Code idea:
```python
def a_star(self, start_position, goal_position):

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
            break

        for successor in self.node_grid.get_successors(node[0], node[1]):
            successor_node = self.node_grid.get_node(successor[0], successor[1])

            g_calculated_cost = node_node.g + self.cost_map.get_edge_cost(node, successor)
            f_calculated_cost = g_calculated_cost + PathPlanner.heuristic(successor, goal_position)

            if successor_node.f > f_calculated_cost:
                if not successor_node.closed:
                    successor_node.g = g_calculated_cost
                    successor_node.f = f_calculated_cost
                    successor_node.parent = node_node
                    heapq.heappush(pq, (successor_node.f, successor))

    path = PathPlanner.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
    cost = self.node_grid.get_node(goal_position[0], goal_position[1]).g

    self.node_grid.reset()
    return path, cost
```

### Greedy Best-first Search

The best-first search is a search algorithm that scans a graph by exploring the most "promising" node according to a specific rule.

Judea Pearl describes the best-first search as the estimation of the quality of a node n by a "heuristic evaluation function f (n) which, in general, may depend on the description of n, of the arrival state, of the information amassed by the algorithm at the time of the evaluation and, most importantly, of additional knowledge about the problem ".

Some authors use the term best-first to refer specifically to a search whose heuristic tries to predict the distance between the end of a path and the solution, so as to explore in priority the path closest to the solution. This specific type of search is named best-first glutton.

> Note:
> A greedy algorithm is an algorithmic paradigm that follows the problem solving heuristic of making the locally optimal choice at each stage with the intent of finding a global optimum. In many problems, a greedy strategy does not usually produce an optimal solution, but nonetheless a greedy heuristic may yield locally optimal solutions that approximate a globally optimal solution in a reasonable amount of time.

Code idea:
```python
def greedy(self, start_position, goal_position):

    pq = []
    node = start_position
    cost = PathPlanner.heuristic(node, goal_position)
    self.node_grid.get_node(node[0], node[1]).f = 0
    heapq.heappush(pq, (cost, node))

    while len(pq) != 0:

        (cost, node) = heapq.heappop(pq)
        node_node = self.node_grid.get_node(node[0], node[1])

        node_node.closed = True

        if node == goal_position:
            break

        for successor in self.node_grid.get_successors(node[0], node[1]):
            successor_node = self.node_grid.get_node(successor[0], successor[1])
            if not successor_node.closed:
                successor_node.f = node_node.f + self.cost_map.get_edge_cost(node, successor)
                successor_node.parent = node_node
                successor_cost = PathPlanner.heuristic(successor, goal_position)
                heapq.heappush(pq, (successor_cost, successor))
                successor_node.closed = True

    path = PathPlanner.construct_path(self.node_grid.get_node(goal_position[0], goal_position[1]))
    cost = self.node_grid.get_node(goal_position[0], goal_position[1]).f
    self.node_grid.reset()
    return path, cost
```

![Alt text](report/imgs/3.png?raw=true "Title")
