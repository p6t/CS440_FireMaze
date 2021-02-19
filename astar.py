import random
import numpy as np
import math
from queue import PriorityQueue as pq

# This file contains implementations for A* (duh)
# Also includes implementation for fire spreading

"""
Euclidean distance metric

Parameters:
first (tuple of (int, int) in x, y order): first point on maze
second (tuple of (int, int) in x, y order): second point on maze

Return:
(float): the distance between both points
"""
def euclidean_heuristic(first, second):
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)

"""
Determine valid neighbors

Parameters:
maze (numpy array): the maze, described in mazegen.py
loc (tuple of (int, int) in x, y order): a cell to find valid neighbors for

Return:
(int): number of valid neigbors of the cell
"""
def valid_neighbors(maze, loc):
    neighbors = []

    x = loc[0]
    y = loc[1]

    if x > 0 and maze[x-1][y] == 0:
        neighbors.append((x - 1, y))
    if x < maze.shape[0] - 1 and maze[x+1][y] == 0:
        neighbors.append((x + 1, y))
    if y > 0 and maze[x][y - 1] == 0:
        neighbors.append((x, y - 1))
    if y < maze.shape[1] - 1 and maze[x][y+1] == 0:
        neighbors.append((x, y + 1))
    
    return neighbors

"""
A* search implementation

Parameters:
maze (numpy array): the maze, described in mazegen.py
start (tuple of (int, int) in x, y order): the start location for the search
goal (tuple of (int, int) in x, y order): the goal location for the search
last_node (dict of (x, y) coordinate tuples to other (x, y) coordinate tuples): last_node[key] gives the node that A* visited before the key node
cur_cost (dict of (x, y): coordinate tuples to ints): cur_cost[key] gives the current cost to travel from start to key

Return:
None
"""
def search(maze, start, goal, last_node, cur_cost):
    
    # Priority queue, woohoo!
    frontier = pq()

    frontier.put(start, 0)
    last_node[start] = start
    cur_cost[start] = 0

    while not frontier.empty():

        cur = frontier.get()

        if cur == goal:
            break

        for next in valid_neighbors(maze, cur):
            new_cost = cur_cost[cur] + 1
            if next not in cur_cost or new_cost < cur_cost[next]:
                cur_cost[next] = new_cost
                priority = new_cost + euclidean_heuristic(next, goal)
                frontier.put(next, priority)
                last_node[next] = cur
    return

"""
A* search implementation, fire included

Parameters:
maze (numpy array): the maze, described in mazegen.py
start (tuple of (int, int) in x, y order): the start location for the search
goal (tuple of (int, int) in x, y order): the goal location for the search
last_node (dict of (x, y) coordinate tuples to other (x, y) coordinate tuples): last_node[key] gives the node that A* visited before the key node
cur_cost (dict of (x, y) coordinate tuples to ints): cur_cost[key] gives the current cost to travel from start to key
q (float between 0 and 1): fire spread chance

Return:
None
"""
def search_with_fire(maze, start, goal, last_node, cur_cost, q):
        
    frontier = pq()

    frontier.put(start, 0)
    last_node[start] = start
    cur_cost[start] = 0

    while not frontier.empty():

        cur = frontier.get()

        if cur == goal:
            break

        for next in valid_neighbors(maze, cur):
            new_cost = cur_cost[cur] + 1
            if next not in cur_cost or new_cost < cur_cost[next]:
                cur_cost[next] = new_cost
                priority = new_cost + euclidean_heuristic(next, goal)
                frontier.put(next, priority)
                last_node[next] = cur

                maze = spread_fire(maze, q)
    return

"""
Spread fire within maze

Parameters:
maze (numpy array): the maze, described in mazegen.py
q (float between 0 and 1): fire spread chance

Return:
(numpy array): the updated maze
"""
def spread_fire(maze, q):
    maze_copy = maze.copy()
    for index, _ in np.ndenumerate(maze):
        # print(index)
        x, y = index
        fire_neighbors = 0
        if maze[x][y] != 1 and maze[x][y] != 2:
            if x != 0:
                if maze[x-1, y] == 2:
                    fire_neighbors += 1
            if y != 0:
                if maze[x, y-1] == 2:
                    fire_neighbors += 1
            if x != maze.shape[0] - 1:
                if maze[x+1, y] == 2:
                    fire_neighbors += 1
            if y != maze.shape[1] - 1:
                if maze[x, y+1] == 2:
                    fire_neighbors += 1
        p_fire = 1 - (1 - q) ** fire_neighbors
        if np.random.random_sample() <= p_fire:
            maze_copy[x][y] = 2
    return maze_copy


