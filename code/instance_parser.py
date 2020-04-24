import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default=None, help='The name of the map file')
    parser.add_argument('--numFiles', type=int, default=1, help='The number of scene files')
    parser.add_argument('--scene', type=str, default=None, help='The prefix of the scene names')
    parser.add_argument('--numAgents', type=int, default=None, help='The number of agents to add')
    args = parser.parse_args()
    # Parse map
    map_name = 'custom_instances/' + args.map
    f = open(map_name, 'r')
    lines = []
    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)

    f.close()
    
    for i in range(1, args.numFiles + 1):

        # Parse scene
        scene_name = 'custom_instances/' + args.scene + str(i) + '.scen'
        f = open(scene_name, 'r')
        agents = []
        for j in range(args.numAgents + 1):
            line = f.readline()
            if not line:
                break
            agent = line.split('\t')
            agents.append(agent)
        
        f.close()
        
        filename = 'custom_instances/instances/instance_' + str(i) + '.txt'
        f = open(filename, 'w')
        height = lines[1].split(' ')[1]
        width = lines[2].split(' ')[1]
        f.write(height[:len(height)-1] + ' ' + width)
        for line in lines[4:]:
            f.write(line)
        f.write(str(len(agents) - 1) + '\n')
        for agent in agents[1:]:
            f.write(agent[5] + ' ' + agent[4] + ' ' + agent[7] + ' ' + agent[6] + '\n')
        f.close()