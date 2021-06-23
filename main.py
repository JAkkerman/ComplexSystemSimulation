import numpy as np

# Import classes
import market
import trader
import visualization as vis
from tqdm import tqdm
import concurrent.futures
import sys
import management
# from cluster import Cluster
from data import avg_degree


def initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc):

    MarketObj = market.Market(p, cluster, garch, garch_param, Pa, Pc)

    for i in range(N_agents):
        TraderObj = trader.Trader(i, MarketObj, A, C)
        MarketObj.traders += [TraderObj]

    # MarketObj.form_pairs()

    return MarketObj


def run_simulation(N_time, MarketObj, cluster):

    for t in tqdm(range(N_time)):
        # if t%1000==0:
            # print('Iteration ', t)

        if cluster:
            MarketObj.form_clusters()
            activated_cluster = MarketObj.activate_cluster()
            # print('yeet', activated_cluster)

        for TraderObj in MarketObj.traders:
            TraderObj.trade_decision()

        # print('yeet sellers', MarketObj.sellers)
        # print('yeet buyers', MarketObj.buyers)

        avg_degree(MarketObj, cluster)

        transaction_q, true_sellers, true_buyers = MarketObj.get_equilibrium_p()
        MarketObj.perform_transactions(transaction_q, true_sellers, true_buyers)
        MarketObj.update_hist_vol()

        if cluster and activated_cluster != None:
            activated_cluster.self_destruct()
        # print(MarketObj.clusters)
        # vis.vis_market_cross(MarketObj)

    #vis.cluster_vis(MarketObj, N_time, cluster)
    #vis.vis_price_series(MarketObj, N_time)
    #vis.vis_wealth_over_time(MarketObj)
    return MarketObj

def job(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i):

    # Loop over every parameter configuration
    for Pa in Pa_list:
        for Pc in Pc_list:

            # Run single simulation for this parameter config
            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc)
            MarketObj = run_simulation(N_time, MarketObj, cluster)
 
            # Save the results form this run
            management.saveSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i, MarketObj)

if __name__ == '__main__':

    N_time = 10000
    N_agents = 100
    cluster = True
    C = 30000
    A = 300
    p = 100

    # Set parameters for Garch
    garch = False
    garch_n = 4
    garch_param = [1,1]

    # Experiment ranges
    Pa_list = [0.0002]
    Pc_list = [0.1]

    # Amount of runs per configuration
    N_concurrent = 2

    # Make directories for each parameter configuration (if they don't exist yet). NB don't comment this out
    management.makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster)

    # Do experiments for all Pa and Pc parameter combinations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(job, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i,) for i in range(N_concurrent)]
    
    # Visualisation single model run 
    visModel = 0
    vis.visualiseSingleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list[0], Pc_list[0], cluster, visModel)

    # Visualisation all model runs of single parameter configuration
    vis.visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list[0], Pc_list[0], cluster, N_concurrent)

    sys.exit()


    for seed in seeds:
        print(f'Seed {seed}')
        np.random.seed(seed)

        # for cluster in [True, False]:
        for cluster in [True, False]:

            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param)
            run_simulation(N_time, MarketObj, cluster)
            Objects.append((MarketObj, cluster))
        # if cluster:
            # vis.cluster_vis(MarketObj, N_time, cluster)
        vis.plot_lorenz_curve(MarketObj)
    # vis.vis_vol_cluster(Objects, 0.2, 10, N_time)
    vis.vis_price_series(Objects, N_time)
    # print(f'Number of sell orders: {len(MarketObj.sellers)}')
    # print(f'Number of buy orders: {len(MarketObj.buyers)}')
