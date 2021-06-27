import numpy as np
from tqdm import tqdm
import concurrent.futures
import sys

# Our own defined modules
import gc
import market
import trader
import visualization as vis
import management
from data import avg_degree


def initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc):
    """
    Initialise simulation run with market and traders.

    @param N_agents             Amount of agents
    @param p                    Initial asset price
    @param C                    Initial cash amount
    @param cluster              Whether clustering is present or not
    @param A                    Initial asset amount
    @param garch                Whether to determine the volatility using GARCH.
    @param garch_n
    @param garch_param
    @param Pa                   Probability of activating a cluster
    @param Pc                   Probability of forming a pair between two agents, i.e clustering probability
    """
    # Initialise market object
    MarketObj = market.Market(p, cluster, garch, garch_param, Pa, Pc)

    # Initialise agents in the market
    for i in range(N_agents):
        TraderObj = trader.Trader(i, MarketObj, A, C)
        MarketObj.traders += [TraderObj]

    return MarketObj


def run_simulation(N_time, MarketObj, cluster):
    """
    Run the Genoa Market Model once.

    @param N_time               Amount of timesteps
    @param MarketObj            Initial market
    @param cluster              Whether clustering is present or not
    """
    for t in tqdm(range(N_time)):

        # Perform clustering steps
        if cluster:
            MarketObj.form_clusters()
            activated_cluster = MarketObj.activate_cluster()

        # All traders place buy or sell order
        for TraderObj in MarketObj.traders:
            TraderObj.trade_decision()

        # Get average clustering degree
        avg_degree(MarketObj, cluster)

        # Determine clearing price and get viable sellers and buyers
        transaction_q, true_sellers, true_buyers = MarketObj.get_equilibrium_p()

        # COMMENT OUT TO VISUALISE DETERMINATION OF CLEARING PRICE
        vis.vis_market_cross(MarketObj, transaction_q)

        # Perform transactions and update the historic volatility
        MarketObj.perform_transactions(transaction_q, true_sellers, true_buyers)
        MarketObj.update_hist_vol()

        # Reset the activated cluster (if there was one)
        if cluster and activated_cluster != None:
            activated_cluster.self_destruct()

    return MarketObj

def job(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i):
    """
    Parallel programming function. It performs a single simulation run for each parameter configuration.

    @param Pa_list              List of cluster activation probabilities
    @param Pc_list              List of clustering probabilities.
    For further parameters, see function getResultsDirectoryPath()
    """
    # Force new seed, otherwise 6 processes (amount of physical cores) will use same seed
    np.random.seed()

    # Loop over every parameter configuration
    for Pa in Pa_list:
        for Pc in Pc_list:
            for N_agent in N_agents:

                # Run single simulation for this parameter config
                MarketObj = initialise(N_agent, p, A, C, cluster, garch, garch_param, Pa, Pc)
                MarketObj = run_simulation(N_time, MarketObj, cluster)

                # Save the results form this run
                management.saveSingleMarket(N_agent, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i, MarketObj)

                # Delete market object
                del MarketObj
                gc.collect()

    return " Done"

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Execute 'python main.py collect' to collect data or 'python main.py visualize' for visualization of stored data.")
        sys.exit()

    # N_time: model iterations, cluster: use agents clustering, C: starting capital of agents, A: starting assets of agents, p: starting asset price.
    N_time = 10000
    cluster = True
    C = 30000
    A = 300
    p = 100

    # Garch parameters
    garch = False
    garch_n = 4
    garch_param = [1,1]

    # Experiment ranges
    Pa_list = [0.0001, 0.0002, 0.0005]
    Pc_list = [0.1, 0.2, 0.3]
    N_agents = [100, 200, 400]

    # Amount of runs per configuration
    N_concurrent = 2 ## NOTE: this was 50 before but this takes a long time
    # number of CPU cores maximally used to collect data
    N_cores = 6

    # Make directories for each parameter configuration (if they don't exist yet). NB don't comment this out
    management.makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster)

    # Do experiments for all Pa and Pc parameter combinations
    if sys.argv[1] == "collect":
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_cores) as executor:
           values = [executor.submit(job, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i,) for i in range(0, N_concurrent)]

    if sys.argv[1] == "visualize":
        # Visualisation all model runs of all parameter configurations
        vis.visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, N_concurrent)
        # Visualisation single model run (first run)
        vis.visualiseSingleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list[1], Pc_list[0], cluster, 0)

    else:
        print("Execute 'python main.py collect' to collect data or 'python main.py visualize' for visualization of stored data.")
        sys.exit()
