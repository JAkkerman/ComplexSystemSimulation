import matplotlib.pyplot as plt
import numpy as np

def vis_market_cross(MarketObj):

    # buy_p = [buyer.b for buyer in MarketObj.buyers]
    # buy_q = [buyer.a_b for buyer in MarketObj.buyers]
    # sell_p = [seller.s for seller in MarketObj.sellers]
    # sell_q = [seller.a_s for seller in MarketObj.sellers]

    sorted_sell = sorted(MarketObj.sellers, key=lambda x: x.s)
    sorted_buy = sorted(MarketObj.buyers, key=lambda x: x.b)[::-1]

    p_sell = [i.s for i in sorted_sell]
    q_sell = np.cumsum([i.a_s for i in sorted_sell])
    p_buy = [i.b for i in sorted_buy]
    q_buy = np.cumsum([i.a_b for i in sorted_buy])

    plt.plot(p_sell, q_sell)
    plt.plot(p_buy, q_buy)
    plt.show()

