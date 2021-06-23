import numpy as np

# Import classes
import market
import trader
import visualization as vis
from tqdm import tqdm
import concurrent.futures
import sys
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

    # vis.cluster_vis(MarketObj, N_time, cluster)
    # vis.vis_price_series(MarketObj, N_time)
    # vis.vis_wealth_over_time(MarketObj)

def job(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc):

    Objects = []

    if len(Pc) == 1 and len(Pa) != 1:
        P_list = Pa
        Pc = Pc[0]
        for Pa in P_list:
            cluster = True
            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc)
            run_simulation(N_time, MarketObj, cluster)
            Objects.append((MarketObj, cluster))

    elif len(Pa) == 1 and len(Pc) != 1:
        P_list = Pc
        Pa = Pa[0]
        for Pc in P_list:
            cluster = True
            MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc)
            run_simulation(N_time, MarketObj, cluster)
            Objects.append((MarketObj, cluster))

    else:
        Pa = Pa[0]
        Pc = Pc[0]
        cluster = True
        MarketObj = initialise(N_agents, p, A, C, cluster, garch, garch_param, Pa, Pc)
        run_simulation(N_time, MarketObj, cluster)
        Objects.append((MarketObj, cluster))



    return Objects


if __name__ == '__main__':

    N_time = 1000
    seeds = [42, 101, 666, 6666, 9000, 12121, 80085, 12345]

    N_agents = 200
    C = 30000
    A = 300
    p = 100

    # Set parameters for Garch
    garch = False
    garch_n = 4
    garch_param = [1,1]

    Pa = [0.0002]
    Pc = [0.1]

    Objects = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        values = [executor.submit(job, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc) for _ in range(10)]

        for f in concurrent.futures.as_completed(values):
            Objects += f.result()

    vis.vis_price_series(Objects, N_time, N_agents)

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
