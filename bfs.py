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

def backtrack(queue,x,y):
    if((x,y) in queue):
        return (queue.index((x,y)) - len(queue)  + 1)
    else:
        return 0

#def fixarrayoffset(queue,offset):
#    for x in range(offset):
#        queue.pop()
#    return queue

#def fixfireoffset(maze,firetrack,offset):
#    track = 0
#    for x,y in firetrack:
#        if(track>=len(firetrack)-offset):
#            maze[x,y] = 0
#            track+=1
#        else:
#            continue
#    return maze








# An implementation of BFS
def BFS_maze(maze, q, goalx, goaly, visited, blocked, queue):
    #offset = abs(backtrack(queue,currentx,currenty))
    while queue:
        
        store = (queue.pop(0))
        todox = store[0]
        todoy = store[1]

        visited.append((todox,todoy))
        #queue = fixarrayoffset(queue,offset)
        if(((todox,todoy)) == ((goalx,goaly))):
            return 1
        
        #store = spread_fire(maze, q)
        #maze = store[0]
        #firetrack.append(store[1])
        #maze = fixfireoffset(maze,firetrack,offset)
        #firetrack = fixarrayoffset(firetrack,offset)
        maze = spread_fire(maze,q)
        #print("\n")
        #print(maze)
        for ex in range(todox-1,todox+2):
            for why in range(todoy-1,todoy+2):
                if(ex <=0 or ex >= maze.shape[0] or why < 0 or why >= maze.shape[1]):
                    continue
                elif ((ex,why)) in visited:
                    continue
                elif((ex,why)) in blocked:
                    continue
                elif(maze[ex,why] != 0):
                    blocked.append((ex,why))
                elif(((ex,why)) in queue):
                    continue
                elif ex == todox or why == todoy:
                    queue.append((ex,why))
                    #return BFS_maze(maze, q, todox, todoy, goalx, goaly, visited, blocked, queue, firetrack)
    
    #end of loop
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

mazedim = 6
maze = start_fire(generate_maze(mazedim, .3))

# PROBLEM 2 TESTING

fire_chance = .1

startx = random.randrange(0, mazedim)
starty = random.randrange(0, mazedim)
goalx = random.randrange(0, mazedim)
goaly = random.randrange(0, mazedim)

print("Starting BFS")

print(maze)
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

print("Starting X:", startx,", StartingY:", starty)
print("Ending X:", goalx,", Ending Y:", goaly)

visited = []
blocked = []
queue = []
#firetrack = []
check = 0
queue.append((startx,starty))
check = BFS_maze(maze, fire_chance, goalx, goaly, visited, blocked, queue)
print("Path:", visited,", Length:", len(visited))
if(check==0):
    print("no path exists")
else:
    print("a path exists")
print("\n")
