import sys
import random
import numpy as np
import astar
import mazegen
import matplotlib.pyplot as plt

# Strategy 1: At the start of the maze, wherever the fire is, solve for the shortest path from upper left to lower right, and follow it until the agent exits the maze or burns.  This strategy does not modify its initial path as the fire changes.
def strat1_astar(mazedim, p, q):

    maze = mazegen.generate_maze(mazedim, p)
    maze = mazegen.start_fire(maze)
    start, goal = (0, 0), (mazedim-1, mazedim-1)

    last_node = {}
    cur_cost = {}

    astar.search(maze, start, goal, last_node, cur_cost)
        
    path = []
    if goal not in last_node:
        # Failure
        return 0
    else:
        cur = goal
        while start not in path:
            path.insert(0, last_node[cur])
            cur = last_node[cur]

    for i in range(len(path)):
        if maze[path[i]] != 0:
            # Fire spread into path
            return 0
        maze = astar.spread_fire(maze, q)

    # Success
    return 1

# Strategy 2: At every time step, re-compute the shortest path from the agent’s current position to the goal position, based on the current state of the maze and the fire.  Follow this new path one time step, then re-compute. This strategy constantly re-adjusts its plan based on the evolution of the fire.  If the agent gets trapped with no path to the goal, it dies.
def strat2_astar(mazedim, p, q):

    maze = mazegen.generate_maze(mazedim, p)
    maze = mazegen.start_fire(maze)
    start, goal = (0, 0), (mazedim-1, mazedim-1)

    cur = start

    while cur != goal:
    
        last_node = {}
        cur_cost = {}
        astar.search(maze, cur, goal, last_node, cur_cost)
            
        if goal not in last_node:
            # Failure
            return 0
        else:
            path = []
            tmp_cur = goal
            while tmp_cur != cur:
                path.insert(0, last_node[tmp_cur])
                if last_node[tmp_cur] == cur:
                    path.append(goal)
                    nxt = path[1]
                tmp_cur = last_node[tmp_cur]
            cur = nxt
        
        maze = astar.spread_fire(maze, q)

    # Success
    return 1

# Strategy 3: At each time step, re-compute the shortest path from the agent's current position to the goal position, based on the worst possible next state of the maze and the fire. If this is impossible, re-compute with the current state of the maze and fire. This strategy accounts for an unknown future by trying harder to avoid coming close to fires, decreasing the chance of being caught.
def strat3_astar(mazedim, p, q):

    maze = mazegen.generate_maze(mazedim, p)
    maze = mazegen.start_fire(maze)
    start, goal = (0, 0), (mazedim-1, mazedim-1)

    cur = start

    while cur != goal:

        # Search with worst possible fire spread
        maze_worstcase = maze.copy()
        maze_worstcase = astar.spread_fire(maze_worstcase, 1)
        last_node = {}
        cur_cost = {}
        astar.search(maze_worstcase, cur, goal, last_node, cur_cost)

        if goal in last_node:
            # Found a path in the worst case
            path = []
            tmp_cur = goal
            while tmp_cur != cur:
                path.insert(0, last_node[tmp_cur])
                if last_node[tmp_cur] == cur:
                    path.append(goal)
                    nxt = path[1]
                tmp_cur = last_node[tmp_cur]
            cur = nxt
            maze = astar.spread_fire(maze, q)
            continue

        # Normal search, if no path in worst case
        last_node = {}
        cur_cost = {}
        astar.search(maze, cur, goal, last_node, cur_cost)


        if goal not in last_node:
            return 0
        else:
            path = []
            tmp_cur = goal
            while tmp_cur != cur:
                path.insert(0, last_node[tmp_cur])
                if last_node[tmp_cur] == cur:
                    path.append(goal)
                    nxt = path[1]
                tmp_cur = last_node[tmp_cur]
            cur = nxt
            maze = astar.spread_fire(maze, q)

    # Success
    return 1

# Set to see full array in console
np.set_printoptions(threshold=sys.maxsize)



mazedim = 10
reps = 1000
p = .3
q_to_test = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

strat1_result = [0] * len(q_to_test)
strat2_result = [0] * len(q_to_test)
strat3_result = [0] * len(q_to_test)

for i in range(len(q_to_test)):
    for j in range(reps):
        strat1_result[i] += strat1_astar(mazedim, p, q_to_test[i])
        strat2_result[i] += strat2_astar(mazedim, p, q_to_test[i])
        strat3_result[i] += strat3_astar(mazedim, p, q_to_test[i])


strat1_result[:] = [x / reps for x in strat1_result]
strat2_result[:] = [x / reps for x in strat2_result]
strat3_result[:] = [x / reps for x in strat3_result]

plt.plot(q_to_test, strat1_result, label = "Strategy 1")
plt.plot(q_to_test, strat2_result, label = "Strategy 2")
plt.plot(q_to_test, strat3_result, label = "Strategy 3")
plt.xlabel("Flammability rate q")
plt.ylabel("Success frequency")
plt.title("Comparison between fire maze strategies, mazedim = " + str(mazedim))
plt.legend()
plt.show()

#print("Strategy 1:", strat1_result, "out of", reps)
#print("Strategy 2:", strat2_result, "out of", reps)
#print("Strategy 3:", strat3_result, "out of", reps)