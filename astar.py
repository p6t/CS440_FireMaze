import random
import numpy as np
import math
from queue import PriorityQueue as pq


def gen_start_and_goal(maze):

    startx = random.randrange(0, maze.shape[0])
    starty = random.randrange(0, maze.shape[1])
    goalx = random.randrange(0, maze.shape[0])
    goaly = random.randrange(0, maze.shape[1])

    while maze[startx, starty] != 0 or maze[goalx, goaly] != 0 or (startx == goalx and starty == goaly):
        if maze[startx, starty] != 0:
            startx = random.randrange(0, maze.shape[0])
            starty = random.randrange(0, maze.shape[1])
        if maze[goalx, goaly] != 0:
            goalx = random.randrange(0, maze.shape[0])
            goaly = random.randrange(0, maze.shape[1])
        if startx == goalx and starty == goaly:
            goalx = random.randrange(0, maze.shape[0])
            goaly = random.randrange(0, maze.shape[1])

    start = (startx, starty)
    goal = (goalx, goaly)

    return (start, goal)

def euclidean_heuristic(first, second):
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)

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

# maze: a numpy array
# start: a tuple (x, y) for maze location
# goal: a tuple (x, y) for maze location
# last_node: a dict linking node to previous step
# cur_cost: a dict linking node to arrival cost from start
def search(maze, start, goal, last_node, cur_cost):
    
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

def spread_fire(maze, q):
    maze_copy = maze
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


