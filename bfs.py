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


# An implementation of BFS
def BFS_maze(maze, q, goalx, goaly, visited, blocked, queue):
    while queue: #while coords still need to be visited

        store = (queue.pop(0)) #pop from the front
        todox = store[0] #coordx
        todoy = store[1] #coordy

        visited.append((todox,todoy)) #append to visited
        if(((todox,todoy)) == ((goalx,goaly))): #if current is the goal
            return 1

        maze = spread_fire(maze,q) #step forward one fire
        #print("\n")
        #print(maze)
        for ex in range(todox-1,todox+2): #for left and right one
            for why in range(todoy-1,todoy+2): #for up and down one
                if(ex <=0 or ex >= maze.shape[0] or why < 0 or why >= maze.shape[1]):
                    continue #if it has exited the maze shape
                elif ((ex,why)) in visited: #if it has already been visited
                    continue
                elif((ex,why)) in blocked: #if the coordinates are blocked
                    continue
                elif(maze[ex,why] != 0): #if the coordinates need to be blocked
                    blocked.append((ex,why))
                elif(((ex,why)) in queue): #if its already in the queue
                    continue
                elif ex == todox or why == todoy: #if it is adjacent to the square
                    queue.append((ex,why)) #append to queue

    else:
        return 0


# Tick fire forward one step
def spread_fire(maze, q):
    maze_copy = maze

    for index, _ in np.ndenumerate(maze):
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
    #store = [maze_copy,((x,y))]
    return maze_copy

# PROBLEM 3

# See bfs.py and astar.py for implementations

"""
BEGIN TESTING CODE
"""

# PROBLEM 1 TESTING


mazedim = 80
maze = start_fire(generate_maze(mazedim, 0.3))

# PROBLEM 2 TESTING

fire_chance = 0

startx = 0
starty = 0
goalx = len(maze)-1
goaly = len(maze)-1

print("Starting BFS")


print(maze)
print("Starting X:", 0 ,", StartingY:", 0)
print("Ending X:", goalx,", Ending Y:", goaly)


visited = []
blocked = []
queue = []
check = 0
queue.append((startx,starty))
check = BFS_maze(maze, fire_chance, goalx, goaly, visited, blocked, queue)
print("Path:", visited,", Length:", len(visited))
if(check==0):
    print("no path exists")
else:
    print("a path exists")
print("\n")
