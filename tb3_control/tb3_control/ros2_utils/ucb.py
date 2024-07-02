from ucb_interfaces.msg import UCBAgentPackage, UCBPackageNode

def encodeloc2id(loc):
    row = loc[0]
    col = loc[1]
    return row * 100 + col

def dict2UCBPackage(node_intervals, node_means):

    package = UCBAgentPackage()
    for node in node_intervals:
        ucbpackage = UCBPackageNode()
        ucbpackage.id = encodeloc2id(node)

        ucbpackage.interval[0] = node_intervals[node][0]
        ucbpackage.interval[1] = node_intervals[node][1]

        ucbpackage.mean = node_means[node]

        package.rewards.append(ucbpackage)

    return package


