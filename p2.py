import sys
import random
import numpy as np
import matplotlib.pyplot as plt


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
def DFS_maze(maze, q, goalx, goaly, visited, blocked, stack):
    while stack:

        store = (stack.pop())
        todox = store[0]
        todoy = store[1]

        visited.append((todox,todoy))
        if(((todox,todoy)) == ((goalx,goaly))):
            return 1


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
                elif(((ex,why)) in stack):
                    continue
                elif ex == todox or why == todoy:
                    stack.append((ex,why))

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
all = []
track = 0
thick = 0.0
while(track<100):
    print(track)
    print(thick)
    mazedim = 30


    maze = start_fire(generate_maze(mazedim, thick))

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

    '''
    print("Path:", visited,", Length:", len(visited))
    if(check==0):
        print("no path exists")
    else:
        print("a path exists")
    '''
    print("\n")
    check*=100
    all.append(check)

    track +=1
    if(track%10==0):
        thick+=0.1

print(all)
n=10
list2 = [sum(all[i:i+n]) /n for i in range(0,len(all),n)]
print("The average per p value from 0 to 0.9: ")
print(list2)



x1 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# plotting the line 1 points
plt.plot(x1, list2)


# naming the x axis
plt.xlabel('Obstacle Density')
# naming the y axis
plt.ylabel('% Chance of Success')
# giving a title to my graph
plt.title('P value vs Success')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
