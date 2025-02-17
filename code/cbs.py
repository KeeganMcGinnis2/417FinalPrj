import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, ida_star, get_CG_cost


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    for i in range( max(len(path1), len(path2)) ):
        if get_location(path1, i) == get_location(path2, i):
            return i, 'v'
        elif get_location(path1, i) == get_location(path2, i+1) and get_location(path1, i+1) == get_location(path2, i):
            return i+1, 'e'

    return None, None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            timestep, collision_type = detect_collision(paths[i], paths[j])
            if timestep:
                if collision_type == 'v':
                    collision = {'a1': i, 'a2': j, 'loc': [get_location(paths[i], timestep)], 'timestep': timestep}
                    collisions.append(collision)
                elif collision_type == 'e':
                    collision = {'a1': i, 'a2': j, 'loc': [get_location(paths[i], timestep-1), get_location(paths[i], timestep)], 'timestep': timestep}
                    collisions.append(collision)
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    if len(collision['loc']) == 1:
        constraint1 = {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']}
        constraint2 = {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep']}
    else:      
        constraint1 = {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']}
        constraint2 = {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep']}
    
    return [constraint1, constraint2]
    

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

    
class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self, iterative=True, CG=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            if iterative:
                path = ida_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                                i, root['constraints'])
            else:
                path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                              i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        while len(self.open_list) > 0:
            p = self.pop_node()
            if len(p['collisions']) == 0:
                self.print_results(p)
                return p['paths']

            collision = p['collisions'][0]
            constraints = standard_splitting(collision)
            for constraint in constraints:
                q = {'cost': 0,'constraints': [], 'paths': [], 'collisions': []}
                q['constraints'] = p['constraints'] + [constraint]
                q['paths'] = p['paths'] + []
                ai = constraint['agent']
                if iterative:
                    path = ida_star(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai],
                                    ai, q['constraints'])
                else:
                    path = a_star(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai],
                                  ai, q['constraints'])
                
                if len(path) > 0:
                    q['paths'][ai] = path
                    q['collisions'] = detect_collisions(q['paths'])
                    if CG:
                        q['cost'] = get_CG_cost(self.my_map, q)
                    else:
                        q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)

        self.print_results(root)
        return 'No solutions'

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

        result_file = open("results.csv", "a", buffering=1)
        cost = get_sum_of_cost(node['paths'])
        result_file.write("{},{},{},{}\n".format(cost, CPU_time, self.num_of_expanded, self.num_of_generated))
        result_file.close()
