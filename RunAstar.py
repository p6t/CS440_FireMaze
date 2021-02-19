import sys
import random
import numpy as np
import astar
import mazegen

# Set to see full array in console
np.set_printoptions(threshold=sys.maxsize)

# PROBLEM 2 TESTING

mazedim = 70

p = 0.3
maze = mazegen.generate_maze(mazedim, p)

visited = []
blocked = []
startx = random.randrange(0, mazedim)
starty = random.randrange(0, mazedim)
goalx = random.randrange(0, mazedim)
goaly = random.randrange(0, mazedim)

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


# PROBLEM 3

# A star

#print("\nStarting A* search:")


#print(maze)

q = 0
start, goal = astar.gen_start_and_goal(maze)
#print("Start:", start, "Goal:", goal)
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
    #print("Total cost:", total_cost)
    print("length: ", len(path))

#print("\n")

