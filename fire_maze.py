import sys
import random
import numpy as np

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

    maze = spread_fire(maze, q)

    if(check == 0):
        if(maze[currenty,currentx] != 0):
            blocked.append((currenty,currentx))
            return 0

        visited.append((currenty,currentx))
        if(currentx==goalx and currenty == goaly):
            check = 1
            return 1

        if(currentx <=0 or currentx >= maze.shape[0] or currenty < 0 or currenty >= maze.shape[1]):
            return 0

        for ex in range(currentx-1,currentx+1):
            for why in range(currenty-1,currenty+1):
                if(check == 1):
                    return 1

                if not (why,ex) in visited:
                    if not (why,ex) in blocked:
                        if ex == currentx or why == currenty:
                            DFS_maze(maze, q, ex, why, goalx, goaly, visited, blocked, check)
    return None

# Tick fire forward one step
def spread_fire(maze, q):
    maze_copy = maze
    for index, _ in np.ndenumerate(maze):
        print(index)
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

# BFS implementation
def BFS_maze():
    pass

# A* implementation
def Astar_maze():
    pass

"""
BEGIN TESTING CODE
"""

# PROBLEM 1 TESTING

mazedim = 4
maze = start_fire(generate_maze(mazedim, .3))

# PROBLEM 2 TESTING

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
