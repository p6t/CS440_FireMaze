import sys
import random
import numpy as np

# Implementations for question 1

"""
Generate new maze

Parameters:
dim (int >= 1): The maze is a square of size dim * dim
p (float between 0 and 1): chance of a given square being a wall

Return:
(numpy array) a dim*dim array representing a maze. check maze[x][y] to see the contents of the maze cell. A value of 0 corresponds to an empty cell. A value of 1 corresponds to a wall. A value of 2 corresponds to a fire.
"""
def generate_maze(dim, p):
    # 0 corresponds to empty space
    maze = np.zeros(dim**2)
    for i in range(int(dim**2 * p)):
        # 1 corresponds to a wall
        maze[i] = 1
    np.random.shuffle(maze)
    maze = maze.reshape((dim, dim))
    maze[0][0] = 0
    maze[dim-1][dim-1] = 0
    return maze

"""
Select initial fire location

Parameters:
maze (numpy array): maze representation, see generate_maze()

Return:
(numpy array): the maze, with a fire at a valid location within
"""
def start_fire(maze):
    x_index = np.random.randint(0, maze.shape[0])
    y_index = np.random.randint(0, maze.shape[1])
    while maze[x_index][y_index] != 0 or (x_index,y_index) == (0, 0) or (x_index, y_index) == (maze.shape[0]-1, maze.shape[1]-1):
        x_index = np.random.randint(0, maze.shape[0])
        y_index = np.random.randint(0, maze.shape[1])
    # 2 corresponds to fire
    maze[x_index][y_index] = 2
    return maze

"""
Tick fire forward one step

Parameters:
maze (numpy array): maze representation, see generate_maze()
q (float between 0 and 1): fire spread chance

Return:
(numpy array): the maze after one time step of fire spread
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