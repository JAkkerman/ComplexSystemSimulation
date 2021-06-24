import pickle
import os
from collections import Mapping, Container
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster):
    return f'results/agents{N_agents}_time{N_time}_C{C}_A{A}_p{p}_garch{garch}_garchN{garch_n}_garchparam{garch_param[0]}{garch_param[1]}_Pa{Pa}_Pc{Pc}_cluster{cluster}'

def makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster):
    # Make directories for each configuration parameter (done in advance due to concurrency problems which otherwise arise)
    for Pa in Pa_list:
        for Pc in Pc_list:
            results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

            if not(os.path.isdir(results_dir)):
                os.mkdir(results_dir)

def saveSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i, MarketObj):
    # Make directory to save results from this simulation
    results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

    # Save results
    with open(f'{results_dir}/{i}.pickle', 'wb') as f:
        pickle.dump(MarketObj, f)

def loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i):
    # Get directory path
    results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

    # Load results
    with open(f'{results_dir}/{i}.pickle', 'rb') as f:
        MarketObj = pickle.load(f)

    return MarketObj

def loadMultipleMarkets(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N):
    # Load all runs
    objects = []
    for i in range(0, N):
        MarketObj = loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i)
        objects.append((MarketObj, [cluster, N_agents, Pa, Pc]))

    return objects
