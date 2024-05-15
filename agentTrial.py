
"""
    The solution for the first homework of the course BE5B33KUI.
    Sources: 1. https://cw.fel.cvut.cz/wiki/_media/courses/be5b33kui/lectures/03_search.pdf 
             2. http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
             3. https://www.youtube.com/watch?v=-L-WgKMFuhE&list=PLFt_AvWsXl0cq5Umv3pMC9SPnKjfp9eGW
    Submitted by: Tejas Bhatnagar 
"""

import kuimaze
import os
import time 
import math
from collections import deque 
import glob
import time 
import csv
import copy
import random
import sys

class AgentAstar(kuimaze.BaseAgent):
    """
        Agent class which inherits from kuimaze.BaseAgent
    """
    def __init__(self, environment):
        self.environment = environment 
        self.closedList = []

    def a_star(self):
        """ 
            Finds the shortest path from the start state to the goal state.
        """
        global start, goal
        observation = self.environment.reset()
        try:
            goal, start = observation[3][0:2], observation[0][0:2]
        except: 
             goal, start = observation[1][0:2], observation[0][0:2]
        startNode = Node(start)
        startNode.gn = 0
        startNode.fn = startNode.gn + startNode.hn
        goalNode = Node(goal)
        openList = PriorityQueue()
        openListPos = [start]
        openList.enqueue(startNode)
        # print("start pos is", startNode.pos)
        # print(startNode.hn)
        # print("goal node is", goalNode.pos)
        # print(goalNode.hn)
        while True:
            """ 
                Psuedo code for this loop. 
                remove currPosition form openList
                add currPosition to closedList
                if currPosition == goal: 
                    break
                for pos in newPositions:
                    if pos is in closedList:
                        continue
                    elif new path (g(n)) to pos is shorter than curr path or pos not in openList:
                        set g(n) to new path
                        set parent to currPosition
                        if pos not in openList:
                            openList.enqueue(pos)
                when outside:
                    backtrack to find the path.
            """ 
            # for n in openList.queue:
            #     print(n.pos, n.hn, n.gn, n.fn, end="\n")
            # print('\n')
            currPosition = openList.dequeue()
            if currPosition == None:
                return None
            openListPos.remove(currPosition.pos)
            self.closedList.append(currPosition.pos)
            # print('hn, gn for this node', currPosition.hn, currPosition.gn)
            # print("new pos are", newPositions)
            if currPosition.pos == goalNode.pos:
                # print('goal reached')
                break
            newPositions = self.environment.expand(currPosition.pos)
            for neighbour in newPositions:
                # print("curr neighbour is", neighbour)
                newPathLength = currPosition.gn + 1
                newPos = neighbour[0]
                if neighbour[1] == 1:
                    if newPos in self.closedList:
                        continue
                    elif newPos not in openListPos:
                        # print('adding this neigh to the open list')
                        openListPos.append(newPos)
                        openList.enqueue(Node(newPos))
                    nodeIndex = openListPos.index(newPos)
                    currNeighbour = openList.queue[nodeIndex]
                    if newPathLength < currNeighbour.gn: 
                        currNeighbour.gn = newPathLength
                        currNeighbour.fn = currNeighbour.gn + currNeighbour.hn
                        currNeighbour.parent = currPosition
                        # print('setting curr neighbour parent to curr position')
                self.environment.render()               # show enviroment's GUI       DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION!      
                # time.sleep(0.1)
        path = []
        currNode = currPosition
        path.append(currNode.pos)
        while currNode != startNode:
            path.append(currNode.parent.pos)
            currNode = currNode.parent
        path.reverse()
        return path


class AgentBFS(kuimaze.BaseAgent):
    """
        An agent class that inherits from kuimaze.BaseAgent and performs Breadth First Search.
    """

    def __init__(self, enviroment):
        self.environment = enviroment
        self.closedList = []
    
    def bfs(self): 
        global start, goal
        observation = self.environment.reset()
        # print(observation)
        try:
            goal, start = observation[3][0:2], observation[0][0:2]
        except: 
             goal, start = observation[1][0:2], observation[0][0:2]
        startNode, goalNode = Node(start), Node(goal)

        queue = Queue()

        queue.push(startNode)
        while not queue.isEmpty(): 
            currPos = queue.pop()
            self.closedList.append(currPos.pos)

            if currPos.pos == goalNode.pos:
                break
            
            newPositions = self.environment.expand(currPos.pos)

            for neigbhour in newPositions:
                newPos = neigbhour[0]

                if neigbhour[1] == 1:
                    # check if already visited 
                    if newPos in self.closedList:
                        continue
                    
                    neighbourNode = Node(newPos)
                    neighbourNode.parent = currPos
                    self.closedList.append(newPos)
                    queue.push(neighbourNode)
                self.environment.render()              
                # time.sleep(0.1)

        # backtrack to find the path
        path = []
        currNode = currPos
        path.append(currNode.pos)
        while currNode != startNode:
            path.append(currNode.parent.pos)
            currNode = currNode.parent
        path.reverse()
        return path


class AgentDFS(kuimaze.BaseAgent):
    """
        An agent class that inherits from kuimaze.BaseAgent and performs Depth First Search. 
    """

    def __init__(self, environment):
        self.environment = environment 
        self.closedList = []


    def dfs(self):
        """
            Class function that performs DFS. 
        """
        global start, goal
        observation = self.environment.reset()

        try:
            goal, start = observation[3][0:2], observation[0][0:2]
        except: 
             goal, start = observation[1][0:2], observation[0][0:2]
        startNode, goalNode = Node(start), Node(goal)
        stack = Stack()
        stack.push(startNode)

        while not stack.isEmpty(): 
            currPos = stack.pop()
            self.closedList.append(currPos.pos)

            if currPos.pos == goalNode.pos:
                break
            
            newPositions = self.environment.expand(currPos.pos)

            for neigbhour in newPositions:
                newPos = neigbhour[0]

                if neigbhour[1] == 1:
                    # check if already visited 
                    if newPos in self.closedList:
                        continue
                    
                    neighbourNode = Node(newPos)
                    neighbourNode.parent = currPos
                    self.closedList.append(newPos)
                    stack.push(neighbourNode)
                self.environment.render()              
                # time.sleep(0.1)

        # backtrack to find the path
        path = []
        currNode = currPos
        path.append(currNode.pos)
        while currNode != startNode:
            path.append(currNode.parent.pos)
            currNode = currNode.parent
        path.reverse()
        return path
        

class Stack():
    """
        A class to implement the stack data structure. 
    """
    def __init__(self):
        self.stack = []

    def push(self, item): 
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def isEmpty(self):
        return len(self.stack) == 0


class Queue(): 
    """
        A class to implement the queue data structure. 
    """

    def __init__(self):
        self.queue = deque()
    
    def push(self, item):
        self.queue.append(item)
    
    def pop(self):
        return self.queue.popleft()
    
    def isEmpty(self):
        return len(self.queue) == 0      


class Node():
    """ 
        A node class which stores some important information about the node.
    """
    def __init__(self, pos):
        self.pos = pos
        self.cost = 1
        self.parent = None
        self.open = False
        self.closed = False
        self.gn = 99999999
        self.hn = heuristics(pos)
        self.fn = self.hn + self.gn
     

class PriorityQueue():
    """
        A priority queue that prioritises elements with least f(n) value.
        This will be reffered to as the "open list" in the find_path function, as it will contain the nodes which are yet to be explored. 
    """
    def __init__(self):
        self.queue = []
    
    def enqueue(self, node):
        """
            Adds the new element to the queue
        """
        self.queue.append(node)
        return
    
    def dequeue(self):
        """
            Removes the element with the least f(n) value.
        """
        sortedQueue = sorted(self.queue, key=lambda x: x.fn, reverse=False)
        try:
            returnElement = sortedQueue.pop(0)
        except IndexError:
            return None
        self.queue.remove(returnElement)
        return returnElement

    
def heuristics(node):
    """
        Finds the heuristics for a given node. 
        The heuristic i've chosen is the  distance from the given node to the goal node
    """
    y, x = node[0], node[1]
    y2, x2 = goal[0], goal[1]
    return math.sqrt((y - y2)**2 + (x - x2)**2)


def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    """
        Finds best policy via value iteration 
        :param problem: {instance} kuimaze.MDPMaze
        :param discount_factor: {int} rate at which the values decay  
        :param epsilon: {int} value for convergence.
        :return: {dict} optimal actions for each state in the format {state: optimal action}
    """

    optimalActionsDict = {}
    valueDict = {}
    states = problem.get_all_states()

    # initialize the value for all states 
    for state in states:
        valueDict[(state.x, state.y)] = 0
        optimalActionsDict[(state.x, state.y)] = None

    while True:
        valueDictCopy = copy.deepcopy(valueDict)        # store values form last iteration 
        delta = 0

        for state in states:

            if not problem.is_terminal_state(state):

                expectedUtility = float('-inf')
                optimalAction = None

                # for each action, evaluate the respective q value
                for action in problem.get_actions(state):
                    
                    qValue = 0   
                    nextStatesAndProbs = problem.get_next_states_and_probs(state, action)

                    for s in nextStatesAndProbs:
                        nextState, probability = s[0], s[1]
                        qValue += probability * valueDictCopy[(nextState.x, nextState.y)]
                    
                    # check if this q value is better than the previous 
                    if qValue > expectedUtility:
                        expectedUtility = qValue
                        optimalAction = action
                
                # update the value for the state 
                valueDict[(state.x, state.y)] = state.reward + discount_factor * (expectedUtility)  # update the values for this iteration 
                optimalActionsDict[(state.x, state.y)] = optimalAction
            else:
                valueDict[(state.x, state.y)] = state.reward
                optimalActionsDict[(state.x, state.y)] = None
            
            diff = abs(valueDict[(state.x, state.y)] - valueDictCopy[(state.x, state.y)])

            if diff > delta:
                delta = diff 

        # break condition 
        if delta < epsilon * ((1 -discount_factor)/discount_factor):
            break

    return optimalActionsDict


def find_policy_via_policy_iteration(problem, discount_factor):
    """
        Finds best policy via policy iteration 
        :param problem: {instance} kuimaze.MDPMaze
        :param discount_factor: {int} rate at which the values decay  
        :return: {dict} optimal actions for each state in the format {state: optimal action}
    """

    optimalActionsDict = {} 
    valueDict = {}
    states = problem.get_all_states()

    # initialize values for each state 
    for state in states:
        if not problem.is_terminal_state(state):
            actions = list(problem.get_actions(state))
            defaultAction = random.choice(actions)

            valueDict[(state.x, state.y)] = 0
            optimalActionsDict[(state.x, state.y)] = defaultAction
        else:
            valueDict[(state.x, state.y)] = 0
            optimalActionsDict[(state.x, state.y)] = None
    
    unchanged = False


    while not unchanged: 
        unchanged = True

        # store values from last iteration
        valueDictCopy = copy.deepcopy(valueDict)
        optimalActionsDictCopy = copy.deepcopy(optimalActionsDict)

        # step 1: value iteration without maxing over all actions, i.e., policy evaluation 
        for state in states:

            if not problem.is_terminal_state(state):

                action = optimalActionsDictCopy[(state.x, state.y)]
                nextStatesAndProbs = problem.get_next_states_and_probs(state, action)
                defaultqValue = 0
                for s in nextStatesAndProbs:
                    nextState, probability = s[0], s[1]
                    reward = problem.get_state_reward(nextState)
                    defaultqValue += probability * (reward + (discount_factor * valueDictCopy[(state.x, state.y)]))

                valueDict[(state.x, state.y)] = defaultqValue

            else:
                valueDict[(state.x, state.y)] = state.reward
                
        
        # step 2: for each state max over actions and find the best one, i.e., policy improvement 
        for state in states: 
            if not problem.is_terminal_state(state):
                actions = problem.get_actions(state)

                expectedUtility = float('-inf')
                optimalAction = None

                for action in actions: 
                    qValue = 0

                    nextStatesAndProbs = problem.get_next_states_and_probs(state, action)

                    for s in nextStatesAndProbs:
                        nextState2, prob = s[0], s[1]
                        reward2 = problem.get_state_reward(nextState2)
                        qValue += prob * (reward2 + (discount_factor * valueDictCopy[(nextState2.x, nextState2.y)]))

                    if qValue > expectedUtility:
                        expectedUtility = qValue
                        optimalAction = action
             

                if expectedUtility > valueDict[(state.x, state.y)]:
                    valueDict[(state.x, state.y)] = expectedUtility
                    optimalActionsDict[(state.x, state.y)] = optimalAction

                if optimalActionsDict[(state.x, state.y)] != optimalActionsDictCopy[(state.x, state.y)]:
                    unchanged = False
            else:
                valueDict[(state.x, state.y)] = state.reward
                optimalActionsDict[(state.x, state.y)] = None


    return optimalActionsDict


def runAllMaps(map_dir, filename): 
    """
        Function to run all the agents and save their solutions. 
    """

    GRAD = (0, 0)
    PROBS = [0.4, 0.3, 0.3, 0]

    results = {'Map': [], 'BFS Times': [], 'DFS Times': [], 'A-star Times': [], 'Value Iteration Times': [], 
               'Policy Iteration Times': [], 'BFS nodes visited': [], 'DFS nodes visited': [], 'A-star nodes visited': [], 
               'BFS path length': [], 'DFS path length': [], 'A-star path length': [],  'Total States': [], 'Total States MDP': []}

    mazes = glob.glob(map_dir)
    # print(mazes)
    for maze in mazes:
        results['Map'].append(maze.split('/')[-1])
        print(f'Map Name: {maze}')
        env = kuimaze.InfEasyMaze(map_image=maze, grad=GRAD)
        env_mdp = kuimaze.MDPMaze(map_image=maze, probs= PROBS, grad=GRAD, node_rewards=None)
        results['Total States'].append(len(env.get_all_states()))
        results['Total States MDP'].append(len(env_mdp.get_all_states()))
        
        agentDFS, agentBFS, agentAstar = AgentDFS(env), AgentBFS(env), AgentAstar(env)

        time_bfs, time_dfs, time_astar, time_value, time_policy = 0, 0, 0, 0, 0
        path_len_bfs, path_len_dfs, path_len_astar = 0, 0, 0
        nodes_bfs, nodes_dfs, nodes_astar = 0, 0, 0

        # average to get most accurate time
        for i in range(10): 
            t1 = time.time()
            path_bfs = agentBFS.bfs()
            t2 = time.time()
            path_len_bfs += len(path_bfs)
            nodes_bfs += len(agentBFS.closedList)
            agentBFS.closedList = []
            time_bfs += t2 - t1

            t3 = time.time()
            path_dfs = agentDFS.dfs()
            t4 = time.time()
            path_len_dfs += len(path_dfs)
            nodes_dfs += len(agentDFS.closedList)
            agentDFS.closedList = []
            time_dfs += t4 - t3

            t5 = time.time()
            path_astar = agentAstar.a_star()
            t6 = time.time()
            path_len_astar += len(path_astar)
            nodes_astar += len(agentAstar.closedList)
            agentAstar.closedList = []
            time_astar += t6 - t5

            t7 = time.time()
            policy_value = find_policy_via_value_iteration(env_mdp, 0.99, 0.03)
            t8 = time.time()
            time_value += t8 - t7
            
            t9 = time.time()
            policy_policy = find_policy_via_policy_iteration(env_mdp, 0.99)
            t10 = time.time()
            time_policy += t10 - t9
        
        time_bfs, time_dfs, time_astar, time_value, time_policy = '{:.6f}'.format(time_bfs / 10), '{:.6f}'.format(time_dfs / 10), '{:.6f}'.format(time_astar / 10), '{:.6f}'.format(time_value / 10), '{:.6f}'.format(time_policy / 10) 
        path_len_bfs, path_len_dfs, path_len_astar = path_len_bfs / 10, path_len_dfs / 10, path_len_astar / 10
        nodes_bfs, nodes_dfs, nodes_astar = nodes_bfs / 10, nodes_dfs / 10 ,nodes_astar / 10

        results['BFS Times'].append(time_bfs), results['DFS Times'].append(time_dfs), results['A-star Times'].append(time_astar), results['Value Iteration Times'].append(time_value), results['Policy Iteration Times'].append(time_policy)
        results['BFS path length'].append(path_len_bfs), results['DFS path length'].append(path_len_dfs), results['A-star path length'].append(path_len_astar)
        results['BFS nodes visited'].append(nodes_bfs), results['DFS nodes visited'].append(nodes_dfs), results['A-star nodes visited'].append(nodes_astar)
    # print(results)
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(results.keys())

        for i in range(len(results['Map'])):
            csv_writer.writerow([results[key][i] for key in results.keys()])
    return 


def sort_files(filein, fileout): 

    with open(filein, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        sorted_rows = sorted(csvreader, key=lambda row: row[0])
    
    with open(fileout, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(sorted_rows)


def get_visualisation_values(dictvalues):
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        # ret.append({'x': key[0], 'y': key[1], 'value': [value, value, value, value]})
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret  


# Driver
if __name__ == '__main__':
    
    GRAD = (0, 0)
    PROBS = [0.4, 0.3, 0.3, 0]
    SAVE_PATH = False
    SAVE_EPS = False
    
    if len(sys.argv) > 1:
        if len(sys.argv) == 2: 
            agent, m = sys.argv[1], 'random'
        else: 
            agent, m = sys.argv[1], sys.argv[2]
        
        if agent == 'astar': 
            if m != "random":
                MAP = f"maps/normal/normal{m}.bmp"
                env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            else: 
                env = kuimaze.InfEasyMaze(map_image=None, grad=GRAD)

            agentAstar = AgentAstar(env)
            path_astar = agentAstar.a_star()
            env.set_path(path_astar)
            env.render(mode='human')
            time.sleep(5)
            
        elif agent == 'bfs': 
            if m != "random":
                MAP = f"maps/normal/normal{m}.bmp"
                env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            else: 
                env = kuimaze.InfEasyMaze(map_image=None, grad=GRAD)

            agentBFS = AgentBFS(env)
            path_bfs = agentBFS.bfs()
            env.set_path(path_bfs)
            env.render(mode='human')
            time.sleep(5)
            
        elif agent == 'dfs': 
            if m != "random":
                MAP = f"maps/normal/normal{m}.bmp"
                env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            else: 
                env = kuimaze.InfEasyMaze(map_image=None, grad=GRAD)
                
            env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            agentDFS = AgentDFS(env)
            path_dfs = agentDFS.dfs()
            env.set_path(path_dfs)
            env.render(mode='human')
            time.sleep(5)

        elif agent == 'value': 
            if m != "random":
                MAP = f"maps/normal/normal{m}.bmp"
                env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            else: 
                env = kuimaze.InfEasyMaze(map_image=None, grad=GRAD)
                
            env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
            env.reset()
            policy1 = find_policy_via_value_iteration(env, 0.999, 0.03)
            env.visualise(get_visualisation_values(policy1))
            env.render()
            time.sleep(5)
        
        elif agent == 'policy': 
            if m != "random":
                MAP = f"maps/normal/normal{m}.bmp"
                env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)
            else: 
                env = kuimaze.InfEasyMaze(map_image=None, grad=GRAD)
                
            env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
            env.reset()
            policy1 = find_policy_via_policy_iteration(env, 0.999)
            env.visualise(get_visualisation_values(policy1))
            env.render()
            time.sleep(5)

        else:
            print("Invalid agent")
