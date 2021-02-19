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
#maze = the actual maze. q is the obstacle Density
#goalx goaly are coordinate goals
# visited marks what has already been visited
#blocked shows what is an obstacle or on fire
#the stack holds what needs to be visited still
def DFS_maze(maze, q, goalx, goaly, visited, blocked, stack):
    while stack: #if any items are left on the stack

        store = (stack.pop()) #pop the last item, so fifo
        todox = store[0] #coordinate x
        todoy = store[1] #coordinate y

        visited.append((todox,todoy)) #append current to visited
        if(((todox,todoy)) == ((goalx,goaly))): #if the current is the goal
            return 1 #exit out


        maze = spread_fire(maze,q) #advance the fire
        #print("\n")
        #print(maze)
        for ex in range(todox-1,todox+2): #for one to the left and one right
            for why in range(todoy-1,todoy+2): #for one up and one down
                if(ex <=0 or ex >= maze.shape[0] or why < 0 or why >= maze.shape[1]):
                    continue #if it has exited the array space
                elif ((ex,why)) in visited: #if it has already been visited
                    continue
                elif((ex,why)) in blocked: #if it has been marked blocked
                    continue
                elif(maze[ex,why] != 0): #if it is on fire or blocked
                    blocked.append((ex,why)) #add to blocked
                elif(((ex,why)) in stack): #if it is already added to the stack
                    continue
                elif ex == todox or why == todoy: #if it is adjacent
                    stack.append((ex,why)) #add to stack

    else:
        return 0


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

mazedim = 85


maze = start_fire(generate_maze(mazedim, 0.3))

# PROBLEM 2 TESTING

#print("Starting DFS")

fire_chance = 0

visited = []
blocked = []
startx = random.randrange(0, mazedim)
starty = random.randrange(0, mazedim)
goalx = random.randrange(0, mazedim)
goaly = random.randrange(0, mazedim)

#print(maze)
#print("loop start")
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

#print("loop end")
check = 0

#print("Starting X:", startx,", StartingY:", starty)
#print("Ending X:", goalx,", Ending Y:", goaly)
stack = [((startx,starty))]
check = DFS_maze(maze, fire_chance, goalx, goaly, visited, blocked, stack)


print("Path:", visited,", Length:", len(visited))
if(check==0):
    print("no path exists")
else:
    print("a path exists")
print("\n")
