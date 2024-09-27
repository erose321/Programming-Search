# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack
    #we use stack because DFS
    stack = Stack()
    #pushes only starting state into the stack - no actions yet since its beginning 
    stack.push((problem.getStartState(), []))
    #keeps track of visted nodes and avoids infinite loops
    visited_nodes = set()

    while not stack.isEmpty():
        curr_state, actions = stack.pop()

        #once pacman makes it to the end, return all actions needed to get there
        if problem.isGoalState(curr_state):
            return actions

        if curr_state not in visited_nodes: 
            visited_nodes.add(curr_state)

            #gets successors of current state along with their action needed to get there and step cost (which is irrelivant here)
            for successor, action, stepCost in problem.getSuccessors(curr_state):
                #we only care about succesors that haven't been visted yet 
                if successor not in visited_nodes:
                    #appends current action to existing list of actions 
                    new_actions = actions + [action]
                    stack.push((successor, new_actions))

    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    from util import Queue

    #since its BFS, we want to use a queue
    queue = Queue()
    #pushes only starting state into the stack
    queue.push((problem.getStartState(), []))
    #keeps track of visted nodes and avoids infinite loops
    visited_nodes = set()

    while not queue.isEmpty():
        curr_state, actions = queue.pop()
        #once pacman makes it to the end, return all actions needed to get there
        if problem.isGoalState(curr_state):
            return actions

        if curr_state not in visited_nodes: 
            visited_nodes.add(curr_state)
            #gets successors of current state along with their action needed to get there and step cost (which is irrelivant here)
            for successor, action, stepCost in problem.getSuccessors(curr_state):
                if successor not in visited_nodes:
                    #appends current action to existing list of actions 
                    new_actions = actions + [action]
                    queue.push((successor, new_actions))

    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    #keep queue for UCS
    p_queue = PriorityQueue()
    #pushes only starting state into the stack
    p_queue.push((problem.getStartState(), [], 0), 0)
    #keeps track of visted nodes and avoids infinite loops
    visited_nodes = set()

    while not p_queue.isEmpty():
        curr_state, actions, cost = p_queue.pop()

        if problem.isGoalState(curr_state):
            return actions

        if curr_state not in visited_nodes: 
            visited_nodes.add(curr_state)
            # The step cost in the tuple is relevant for this problem (ufs) since it uses "shortest path" logic or "least cost" in this case
            for successor, action, step_cost in problem.getSuccessors(curr_state):
                if successor not in visited_nodes:
                    #appends current action to existing list of actions 
                    new_actions = actions + [action]
                    total_cost = step_cost + cost
                    #the "priority" includes the total cost 
                    p_queue.push((successor, new_actions, total_cost), total_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    
    """
    Search the node that has the lowest combined cost and heuristic first.
    
    "*** YOUR CODE HERE ***"
    """

    from util import PriorityQueue
    #since we care about path costs, we will continue to use a pqueue here and we will call it the fringe as we used in class
    fringe = PriorityQueue()
    #push initial state on to pqueue which includes an empty list of actions, a cost of 0 since we haven't gone anywhere yet, 
    #and a "priority" that is equal to the given heuristic value
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
    #keeps track visited nodes and the costs that it took to reach those nodes
    visited_nodes = {}

    while not fringe.isEmpty():
        curr_state, actions, cost = fringe.pop()

        if problem.isGoalState(curr_state):
            return actions

        # If the current state hasn't been visited or we found a cheaper path to it, mark it as visited
        # and store the cost that it took to reach this state
        if curr_state not in visited_nodes or cost < visited_nodes[curr_state]:
            visited_nodes[curr_state] = cost

            for successor, action, step_cost in problem.getSuccessors(curr_state):
                new_actions = actions + [action]
                # calculates cumiltive cost to reach successor 
                new_cost = cost + step_cost 
                #f(n) g(n) + h(n) equation. This is the value that a* uses to figure out which path to take. its the sum of the backwards and forwards cost
                sum_backwards_forwards = new_cost + heuristic(successor, problem)
                # If the successor hasn't been visited or we found a cheaper path to it, push it to p queue 
                # with the updated total cost and priority f(n).
                if successor not in visited_nodes or new_cost < visited_nodes[successor]:
                     fringe.push((successor, new_actions, new_cost), sum_backwards_forwards) 
                    

    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
