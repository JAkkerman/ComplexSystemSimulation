import numpy as np

# Import classes
import market
import trader
import visualization as vis
# from cluster import Cluster


def initialise(N_agents, p, A, C, cluster):

    MarketObj = market.Market(p, cluster)

    for i in range(N_agents):
        TraderObj = trader.Trader(i, MarketObj, A, C)
        MarketObj.traders += [TraderObj]

    MarketObj.form_pairs()

    return MarketObj


def run_simulation(N_time, MarketObj):

    for t in range(N_time):
        if t%1000==0:
            print('Iteration ', t)
        for TraderObj in MarketObj.traders:
            TraderObj.trade_decision()

        # print('yeet sellers', MarketObj.sellers)
        # print('yeet buyers', MarketObj.buyers)
        
        transaction_q, true_sellers, true_buyers = MarketObj.get_equilibrium_p()
        MarketObj.perform_transactions(transaction_q, true_sellers, true_buyers)
        MarketObj.update_hist_vol()
        # MarketObj.form_clusters()
        # vis.vis_market_cross(MarketObj)

    vis.vis_price_series(MarketObj)
    # vis.vis_wealth_over_time(MarketObj)


if __name__ == '__main__':

    N_time = 10000
    N_agents = 100
    C = 30000
    A = 300
    p = 100
    cluster = True

    MarketObj = initialise(N_agents, p, A, C, cluster)
    run_simulation(N_time, MarketObj)

    # print(f'Number of sell orders: {len(MarketObj.sellers)}')
    # print(f'Number of buy orders: {len(MarketObj.buyers)}')
