import numpy as np
import math

class Trader():
    def __init__(self, id, market, A, C):
        self.id = id
        self.market = market # add market object for easy access to parameters
        self.C = [C] # initialize list for cash time series
        self.A = [A] # initialize list for stock time series
        self.a_b = 0 # quantity of stocks to buy
        self.b_i = 0 # limit buy price of trader's buy order
        self.a_s = 0 # quantity of stock offered for sale
        self.s_i = 0 # limit sell price of trader's sell order
        self.P_i = 0.5 # buy order probability
        self.in_cluster = None

    def trade_decision(self): # determine whether buyer or seller
        # print(self.id)
        if np.random.random() > self.P_i:
            self.market.buyers += [self]
            self.buy()
        else:
            self.market.sellers += [self]
            self.sell()

    def buy(self):
        # print(f'agent {self.id} buys')

        # print(self.market.mu, self.market.sigma)

        b_i = max(0, self.market.p[-1]*np.random.normal(self.market.mu, self.market.sigma))
        # print(f'b_i = {b_i}')

        # Generate buy limit price
        if b_i == 0:
            self.a_b = 0
            self.b_i = 0
        else:
            a_b = max(0, math.trunc(np.random.random()*self.C[-1]/b_i))

            self.b_i = b_i
            self.a_b = a_b

    def sell(self):
        # print(f'agent {self.id} sells')

        # print('assets agent ', self.id, self.A[-1])
        a_s = max(0, math.trunc(np.random.random()*self.A[-1]))

        # print('agent wants to sell', a_s, k)

        # Generate sell limit price
        s_i = max(0, self.market.p[-1]/np.random.normal(self.market.mu, self.market.sigma))
        # print(f's_i = {s_i}')

        self.s_i = s_i
        self.a_s = a_s

    def no_trade(self):
        """
        Called when no trade is made, in order to keep the asset and cash series
        consistent
        """
        self.A += [self.A[-1]]
        self.C += [self.C[-1]]
