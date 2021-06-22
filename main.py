import numpy as np

# Import classes
import market
import trader
import visualization as vis
from tqdm import tqdm
# from cluster import Cluster
from data import avg_degree


def initialise(N_agents, p, A, C, cluster):

    MarketObj = market.Market(p, cluster)

    for i in range(N_agents):
        TraderObj = trader.Trader(i, MarketObj, A, C)
        MarketObj.traders += [TraderObj]

    MarketObj.form_pairs()

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


if __name__ == '__main__':

    N_time = 50000
    N_agents = 100
    C = 30000
    A = 300
    p = 100
    # cluster = True
    Objects = []
    for cluster in [True, False]:

        MarketObj = initialise(N_agents, p, A, C, cluster)
        run_simulation(N_time, MarketObj, cluster)
        Objects.append((MarketObj, cluster))
        # if cluster:
            # vis.cluster_vis(MarketObj, N_time, cluster)
    vis.vis_vol_cluster(Objects, 0.2, 10, N_time)
    # vis.vis_price_series(Objects)
    # print(f'Number of sell orders: {len(MarketObj.sellers)}')
    # print(f'Number of buy orders: {len(MarketObj.buyers)}')
