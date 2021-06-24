import numpy as np
from tqdm import tqdm
import concurrent.futures
import sys

# Our defined modules
import gc
import market
import trader
import visualization as vis
import management
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

        # vis.vis_market_cross(MarketObj, transaction_q)

        MarketObj.perform_transactions(transaction_q, true_sellers, true_buyers)
        MarketObj.update_hist_vol()

        if cluster and activated_cluster != None:
            activated_cluster.self_destruct()
        # print(MarketObj.clusters)

        

    #vis.cluster_vis(MarketObj, N_time, cluster)
    #vis.vis_price_series(MarketObj, N_time)
    #vis.vis_wealth_over_time(MarketObj)
    return MarketObj

def job(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i):

    # Force new seed, otherwise 6 processes will use same seed
    #print(np.random.uniform(0, 1, 5))
    np.random.seed()
    #print(np.random.uniform(0, 1, 5))

    # Loop over every parameter configuration
    for Pa in Pa_list:
        for Pc in Pc_list:

            # Run single simulation for this parameter config
            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc)
            MarketObj = run_simulation(N_time, MarketObj, cluster)

            # Save the results form this run
            management.saveSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i, MarketObj)
            #print("Size:")
            #print(management.get_size(MarketObj)) --> 80mb

            del MarketObj
            gc.collect()

    return " Done"

if __name__ == '__main__':

    N_time = 10002
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
    Pa_list = [0.0001, 0.0002, 0.0005]
    Pc_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Amount of runs per configuration
    N_concurrent = 2

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     values = [executor.submit(job, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc) for _ in range(10)]

    #     for f in concurrent.futures.as_completed(values):
    #         Objects += f.result()

    # print(Objects)
    # sys.exit()
    # Make directories for each parameter configuration (if they don't exist yet). NB don't comment this out
    # management.makeDirectories(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster)

    # Do experiments for all Pa and Pc parameter combinations
    #with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    #    values = [executor.submit(job, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, i,) for i in range(0, N_concurrent)]

        #for f in concurrent.futures.as_completed(values):
        #    print(f.result())

    # Visualisation single model run
    #for i in range(10,  N_concurrent):
    #    vis.visualiseSingleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list[1], Pc_list[1], cluster, i)

    # for seed in seeds:
    #     print(f'Seed {seed}')
    #     np.random.seed(seed)


    # FOR VIS OF LORENZ OVER TIME, CAN BE DELETED LATER
    # for cluster in [True, False]:
    # objects = []
    # for i in range(10):
    # # for N_agents in [100, 200, 400]:
    #     for cluster in [True]:

    #         MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa_list[1], Pc_list[0])
    #         run_simulation(N_time, MarketObj, cluster)
    #         # Objects.append((MarketObj, cluster))
    #     # if cluster:
    #         # vis.cluster_vis(MarketObj, N_time, cluster)
    #         # vis.plot_wealth_dist(MarketObj)
    #         objects += [MarketObj]
    # vis.plot_lorenz_curve(objects)
    # vis.plot_lorenz_curve_Nagents(objects)

    # FOR VIS OF LORENZ FOR DIFFERENT AMOUNTS, CAN BE DELETED LATER
    # all_N_agents = [100, 200, 400]
    objects = []
    for i in range(10):
        # for N_agents in all_N_agents:
        for cluster in [True]:

            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa_list[1], Pc_list[0])
            run_simulation(N_time, MarketObj, cluster)
            # Objects.append((MarketObj, cluster))
        # if cluster:
            # vis.cluster_vis(MarketObj, N_time, cluster)
            vis.plot_wealth_dist(MarketObj)
            objects += [MarketObj]
    # vis.plot_lorenz_curve_Nagents(objects, all_N_agents)


    # vis.vis_vol_cluster(Objects, 0.2, 10, N_time)
    # vis.vis_price_series(Objects, N_time)
    # print(f'Number of sell orders: {len(MarketObj.sellers)}')
    # print(f'Number of buy orders: {len(MarketObj.buyers)}')
    # Visualisation all model runs of single parameter configuration
    # vis.visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list[0], Pc_list[0], cluster, N_concurrent)
    # for Pa in Pa_list:
    #     for Pc in Pc_list:
    #         vis.visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N_concurrent)

    # sys.exit()

