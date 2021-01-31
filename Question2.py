import sys
import random
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

mazedim = 4


#problem 1
def generate_maze(dim, p):
    maze = np.zeros(dim**2)
    for i in range(int(dim**2 * p)):
        maze[i] = 1
    np.random.shuffle(maze)
    return maze.reshape((dim, dim))

maze = generate_maze(mazedim, .3)


#problem 2
visited = []
blocked = []
startx = random.randrange(0, mazedim)
starty = random.randrange(0, mazedim)
goalx = random.randrange(0, mazedim)
goaly = random.randrange(0, mazedim)

print(maze)
print("loop start")
while maze[startx,starty] ==1 or maze[goalx,goaly]==1 or (startx == goalx and starty == goaly):
    if maze[starty,startx] ==1:
        startx = random.randrange(0, mazedim)
        starty = random.randrange(0, mazedim)
    if maze[goaly,goalx]==1:
        goalx = random.randrange(0, mazedim)
        goaly = random.randrange(0, mazedim)
    if startx == goalx and starty == goaly:
        goalx = random.randrange(0, mazedim)
        goaly = random.randrange(0, mazedim)

print("loop end")
check = 0

def DFS_maze(maze, currentx, currenty, goalx, goaly, visited, blocked, check):

    if(check ==0):
        #DFS Iimplementation
        #add the current square to avoid repeats
        if(maze[currenty,currentx]==1):
            blocked.append((currenty,currentx))
            return 0
        visited.append((currenty,currentx))

        if(currentx==goalx and currenty == goaly):
            check = 1
            return 1
        #if the square is blocked, you cannot go to it
        
        if(currentx<=0 or currentx>=mazedim or currenty<0 or currenty>=mazedim):
            return 0
        for ex in range(currentx-1,currentx+1):
            for why in range(currenty-1,currenty+1):
                if(check ==1):
                    return 1
                if not (why,ex) in visited:
                    if not (why,ex) in blocked:
                        if ex == currentx or why == currenty:
                            DFS_maze(maze, ex, why, goalx, goaly, visited, blocked, check)

print("Starting X: ",startx," Starting Y: ",starty)
print("Ending X: ",goalx," Ending Y: ",goaly)
check = DFS_maze(maze,startx,starty,goalx,goaly,visited, blocked, check)



print("Path: ", visited,"Length: ", len(visited))
if(check==0):
    print("no path exists")
else:
    print("a path exists")
