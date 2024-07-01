from ucb_interfaces.msg import UCBAgentPackage, UCBPackageNode

def dict2UCBPackage(node_intervals, node_means):

    package = UCBAgentPackage()
    for node in node_intervals:
        ucbpackage = UCBPackageNode()
        ucbpackage.loc[0] = node[0]
        ucbpackage.loc[1] = node[1]

        ucbpackage.interval[0] = node_intervals[node][0]
        ucbpackage.interval[1] = node_intervals[node][1]

        ucbpackage.mean = node_means[node]

        package.rewards.append(ucbpackage)

    return package


