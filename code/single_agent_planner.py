import heapq
import math
import time

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def get_CG_cost(my_map, N):
    h = 0
    #c for agent 0,  c = len(N['paths'][0])
    MDDs = []
    numberOfAgents = len(N['paths'])
    print("number of agents", numberOfAgents)
    for agent in range(numberOfAgents):
        c = len(N['paths'][agent])
        MDD = get_MDD(my_map, N['paths'][agent][0], N['paths'][agent][-1], agent, N['constraints'], c)
        MDDs.append(MDD)
        print_MDD(MDD)
    return get_sum_of_cost(N['paths']) + h


def print_MDD(MDD):
    print("MDD:")
    for location in MDD:
        print(location)
        

def get_MDD(my_map, start_loc, goal_loc, agent, constraints, c):
    MDD = [set() for i in range(c)]
    open_list = []
    #closed_list = dict()
    constraint_table, max_constraint_time = build_constraint_table(constraints, agent)
    upper_bound = c
    root = {'loc': start_loc, 'g_val': 0, 'h_val': 0, 'parent': None, 'timestep': 0}
    push_node(open_list, root, 0)
    #closed_list[(root['loc'], root['timestep'])] = root
    nodeCount = 0
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #path = get_path(curr)
        #print(path)
        if curr['timestep'] >= upper_bound:
            return MDD
        if curr['loc'] == goal_loc and curr['timestep'] >= max_constraint_time and c-1 == curr['timestep']:
            path = get_path(curr)
            #print(path)
            for i in range(c):
                MDD[i].add(path[i])
            continue
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] == -1 or child_loc[0] == len(my_map) or child_loc[1] == -1 or child_loc[1] == len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]] or is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': 0,
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
            #if (child['loc'], child['timestep']) in closed_list:
             #   existing_node = closed_list[(child['loc'], child['timestep'])]
              #  if compare_nodes(child, existing_node):
                    #closed_list[(child['loc'], child['timestep'])] = child
               #     push_node(open_list, child)
           # else:
                #closed_list[(child['loc'], child['timestep'])] = child
            push_node(open_list, child, nodeCount)
            nodeCount += 1
    print(agent)
    time.sleep(1)
    return None  # Failed to find solutions

def get_cardinal_conflicts(MDDs, numberOfAgents, c):
    cardinal_conflicts = []
    for i in range(numberOfAgents):
        for j in range(i, numberOfAgents):
            ci = len(MDDs[i])
            for timestep in range(ci):
                if(len(MDDs[i]) == 1 and len(MDDs[j]) == 1 and MDDs[i] == MDDs[j]):
                    cardinal.conflicts({
                        'agent1': i,
                        'agent2': j,
                    })
                    break

def mvc(cardinal_conflicts):
    #fuck
    pass

def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that contains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    constraint_table = {}
    max_timestep = 0
    for constraint in constraints:
        if constraint['agent'] == agent:
            if constraint['timestep'] in constraint_table:
                constraint_table[constraint['timestep']].append(constraint)
            else:
                constraint_table[constraint['timestep']] = [constraint]

            if constraint['timestep'] > max_timestep:
                max_timestep = constraint['timestep']

    return constraint_table, max_timestep
    

def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            if [next_loc] == constraint['loc']:
                return True
            # Handle edge constraints
            elif [curr_loc, next_loc] == constraint['loc']:
                return True
    
        return False
    else:
        return False


def push_node(open_list, node, nodeCount):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], nodeCount, node))


def pop_node(open_list):
    _, _, _, _, curr = heapq.heappop(open_list)
    return curr

def compare_nodes(n1, n2):
    """Return true if n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict()
    constraint_table, max_constraint_time = build_constraint_table(constraints, agent)
    upper_bound = len(my_map)*len(my_map[0])
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
    push_node(open_list, root, 0)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['timestep'] > upper_bound:
            return None
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and curr['timestep'] >= max_constraint_time:
            return get_path(curr)
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] == -1 or child_loc[0] == len(my_map) or child_loc[1] == -1 or child_loc[1] == len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]] or is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child, 0)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child, 0)

    return None  # Failed to find solutions


def ida_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    # •A cost threshold l is set.
    # •f(n) = g(n) + h(n) is computed in each iteration.
    # •If f(n) < l we expand the node.
    # •Else the branch is pruned (we don’t expand it).
    # •If a goal node is reached with a cost lower than the threshold, then the goal it is returned.
    # •Else if a whole iteration has ended without reaching the goal,then another iteration is begun 
    # with a greater cost threshold.
    # •The new cost threshold is set to the minimum cost of all nodes that were pruned on the previous
    # iteration.
    # •The cost Threshold for the first Iteration is set to the cost of the initial state.
    h_value = h_values[start_loc]
    threshold = h_value
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}
    constraint_table, max_constraint_time = build_constraint_table(constraints, agent)
    while True:
        t = search(my_map, root, goal_loc, h_values, agent, constraint_table, max_constraint_time, threshold)
        if type(t) == dict:
            # print('complete')
            # print(agent, '   ', get_path(t))
            # time.sleep(1)
            return get_path(t)
        elif t == math.inf:
            # print('fail')
            # time.sleep(1)
            return None
        threshold = t


def search(my_map, curr, goal_loc, h_values, agent, constraint_table, max_constraint_time, threshold):
    f = curr['g_val'] + curr['h_val']
    if f > threshold:
        return f
    elif curr['loc'] == goal_loc and curr['timestep'] >= max_constraint_time:
        return curr
    minimum = math.inf
    for dir in range(5):
        child_loc = move(curr['loc'], dir)
        if child_loc[0] == -1 or child_loc[0] == len(my_map) or child_loc[1] == -1 or child_loc[1] == len(my_map[0]):
            continue
        if my_map[child_loc[0]][child_loc[1]] or is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
            continue
        child = {'loc': child_loc,
                'g_val': curr['g_val'] + 1,
                'h_val': h_values[child_loc],
                'parent': curr,
                'timestep': curr['timestep'] + 1}
        t = search(my_map, child, goal_loc, h_values, agent, constraint_table, max_constraint_time, threshold)
        if type(t) == dict:
            return t
        elif t < minimum:
            minimum = t
    return minimum