import sys
import random
import numpy as np
import astar
import bfs

# Set to see full array in console
np.set_printoptions(threshold=sys.maxsize)

# PROBLEM 1

# Generate a maze
def generate_maze(dim, p):
    # 0 corresponds to empty space
    maze = np.zeros(dim**2)
    for i in range(int(dim**2 * p)):
        # 1 corresponds to a wall
        maze[i] = 1
    np.random.shuffle(maze)
    return maze.reshape((dim, dim))

# Select an initial fire location
def start_fire(maze):
    x_index = np.random.randint(0, maze.shape[0])
    y_index = np.random.randint(0, maze.shape[1])
    while maze[x_index][y_index] != 0:
        x_index = np.random.randint(0, maze.shape[0])
        y_index = np.random.randint(0, maze.shape[1])
    # 2 corresponds to fire
    maze[x_index][y_index] = 2
    return maze

# PROBLEM 2

# An implementation of DFS
def DFS_maze(maze, q, currentx, currenty, goalx, goaly, visited, blocked, check):

    if(currentx <=0 or currentx >= maze.shape[0] or currenty < 0 or currenty >= maze.shape[1]):
            return 0

    maze = spread_fire(maze, q)

    if(check == 0):
        if(maze[currenty,currentx] != 0):
            blocked.append((currenty,currentx))
            return 0

        visited.append((currenty,currentx))
        if(currentx==goalx and currenty == goaly):
            check = 1
            return 1

        for ex in range(currentx-1,currentx+2):
            for why in range(currenty-1,currenty+2):
                if(check == 1):
                    return 1

                if not (ex,why) in visited:
                    if not (ex,why) in blocked:
                        if ex == currentx or why == currenty:
                            DFS_maze(maze, q, ex, why, goalx, goaly, visited, blocked, check)
    return 1

# Tick fire forward one step
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

# PROBLEM 3

# See bfs.py and astar.py for implementations

"""
BEGIN TESTING CODE
"""

# PROBLEM 1 TESTING

mazedim = 4
maze = start_fire(generate_maze(mazedim, .3))

# PROBLEM 2 TESTING

print("Starting DFS")

fire_chance = .1

visited = []
blocked = []
startx = random.randrange(0, mazedim)
starty = random.randrange(0, mazedim)
goalx = random.randrange(0, mazedim)
goaly = random.randrange(0, mazedim)

print(maze)
print("loop start")
while maze[startx, starty] != 0 or maze[goalx, goaly] != 0 or (startx == goalx and starty == goaly):
    if maze[startx, starty] != 0:
        startx = random.randrange(0, mazedim)
        starty = random.randrange(0, mazedim)
    if maze[goalx, goaly] != 0:
        goalx = random.randrange(0, mazedim)
        goaly = random.randrange(0, mazedim)
    if startx == goalx and starty == goaly:
        goalx = random.randrange(0, mazedim)
        goaly = random.randrange(0, mazedim)

print("loop end")
check = 0

print("Starting X:", startx,", StartingY:", starty)
print("Ending X:", goalx,", Ending Y:", goaly)
check = DFS_maze(maze, fire_chance, startx, starty, goalx, goaly, visited, blocked, check)


print("Path:", visited,", Length:", len(visited))
if(check==0):
    print("no path exists")
else:
    print("a path exists")
print("\n")

# PROBLEM 3

# A star

print("\nStarting A* search:")

mazedim = 6
p = .3
maze = generate_maze(mazedim, p)
print(maze)

q = .1
start, goal = astar.gen_start_and_goal(maze)
print("Start:", start, "Goal:", goal)
last_node = {}
cur_cost = {}

astar.search_with_fire(maze, start, goal, last_node, cur_cost, q)

path = []
if goal not in last_node:
    print("No path found.")
else:
    total_cost = cur_cost[goal]
    cur = goal
    while start not in path:
        path.insert(0, last_node[cur])
        cur = last_node[cur]
    print("Total cost:", total_cost)
    print("Path:", path)
print("\n")
