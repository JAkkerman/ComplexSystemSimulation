import pickle
import os

def getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc):
    return f'results/agents{N_agents}_time{N_time}_C{C}_A{A}_p{p}_garch{garch}_garchN{garch_n}_garchparam{garch_param[0]}{garch_param[1]}_Pa{Pa}_Pc{Pc}'

def makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list):
    for Pa in Pa_list:
        for Pc in Pc_list:
            results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc)
            if not(os.path.isdir(results_dir)):
                os.mkdir(results_dir)

def saveSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, i, MarketObj):
    # Make directory to save results from this simulation
    results_dir = getResultsDirectoryPath(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc)

    # Save results
    with open(f'{results_dir}/{i}.pickle', 'wb') as f:
        pickle.dump(MarketObj, f)

def loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, i):
    # Get directory path
    results_dir = getResultsDirectoryPath(
        N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc)

    # Load results
    with open(f'{results_dir}/{i}.pickle', 'rb') as f:
        MarketObj = pickle.load(f)

    return MarketObj
