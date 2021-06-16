import numpy as np

# Import classes
import market
import trader
import visualization as vis
# from cluster import Cluster


def initialise(N_agents, p, A, C):

    MarketObj = market.Market(p)

    for i in range(N_agents):
        TraderObj = trader.Trader(i, MarketObj, A, C)
        MarketObj.traders += [TraderObj]

    return MarketObj


def run_simulation(N_time, MarketObj):

    for t in range(N_time):
        for TraderObj in MarketObj.traders:
            TraderObj.trade_decision()
        vis.vis_market_cross(MarketObj)
        # MarketObj.get_equilibrium_p()


if __name__ == '__main__':

    N_time = 1
    N_agents = 100
    C = 30000
    A = 300
    p = 100

    MarketObj = initialise(N_agents, p, A, C)
    run_simulation(N_time, MarketObj)

    print(f'Number of sell orders: {len(MarketObj.sellers)}')
    print(f'Number of buy orders: {len(MarketObj.buyers)}')
