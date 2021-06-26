import pickle
import os
from collections import Mapping, Container
import sys


def getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster):
    """"
    Returns the directory of the results of a specific run configuration.

    @param N_agents             Amount of agents
    @param N_time               Amount of timesteps
    @param C                    Initial cash amount
    @param A                    Initial asset amount
    @param p                    Initial asset price
    @param garch                Whether to determine the volatility using GARCH.
    @param garch_n          
    @param garch_param
    @param Pa                   Probability of activating a cluster
    @param Pc                   Probability of forming a pair between two agents, i.e clustering probability
    @param cluster              Whether clustering is present or not
    """
    return f'results/agents{N_agents}_time{N_time}_C{C}_A{A}_p{p}_garch{garch}_garchN{garch_n}_garchparam{garch_param[0]}{garch_param[1]}_Pa{Pa}_Pc{Pc}_cluster{cluster}'


def makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster):
    """
     Make directories for each configuration parameter (done in advance due to concurrency problems which otherwise arise)

     @param Pa_list             List of cluster activation probabilities
     @param Pc_list             List of clustering probabilities.
     For further parameters, see function getResultsDirectoryPath()
    """

    # Loop over all possible parameter configurations
    for Pa in Pa_list:
        for Pc in Pc_list:
            # Get the directory of this parameter configuration
            results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

            # Make the directory if it doesn't exist yet
            if not(os.path.isdir(results_dir)):
                os.mkdir(results_dir)


def saveSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i, MarketObj):
    """
    Write the final MarketObject to disk.

    @param i                    The run number (i.e, 0 to 49)
    @param MarketObj            The MarketObj at the end of the run
    For further parameters, see function getResultsDirectoruPath()
    """
    
    #Make directory to save results from this simulation
    results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

    # Save results
    with open(f'{results_dir}/{i}.pickle', 'wb') as f:
        pickle.dump(MarketObj, f)


def loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i):
    """
    Load a MarketObject from disk.

    @param i                    The run number (i.e, 0 to 49)
    For further parameters, see function getResultsDirectoruPath()
    """

    # Get directory path
    results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster)

    # Load results
    with open(f'{results_dir}/{i}.pickle', 'rb') as f:
        MarketObj = pickle.load(f)

    return MarketObj


def loadMultipleMarkets(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N):
    """
    Load all N MarketObjects from a single parameter configuration.

    @param N                    The amount of simulation results.
    For further parameters, see function getResultsDirectoruPath()
    """

    # Load N runs
    objects = []
    for i in range(0, N):
        MarketObj = loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i)
        objects.append((MarketObj, [cluster, N_agents, Pa, Pc]))

    return objects
