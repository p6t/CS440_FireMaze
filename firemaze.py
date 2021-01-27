import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def generate_maze(dim, p):
    
    maze = np.zeros(dim**2)

    for i in range(int(dim**2 * p)):
        maze[i] = 1
    
    np.random.shuffle(maze)
    
    return maze.reshape((dim, dim))

testmaze = generate_maze(3, .3)
print(testmaze)
