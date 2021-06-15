import numpy as np
import math

class Trader():
    def __init__(self, id, market, A, C):
        self.id = id
        self.market = market
        self.C = [C]
        self.A = [A]
        self.a_b = 0
        self.b = 0
        self.a_s = 0
        self.s = 0
        self.P = 0.5

    def trade_decision(self):
        if np.random.random() > 0.5:
            self.market.buyers += [self]
            self.buy()
        else:
            self.market.sellers += [self]
            self.sell()

    def buy(self):
        # print(f'agent {self.id} buys')

        # print(self.market.mu, self.market.sigma)

        b = self.market.p[-1]*np.random.normal(self.market.mu, self.market.sigma)

        a_b = math.trunc(np.random.random()*self.C[-1]/b)

        self.b = b
        self.a_b = a_b

    def sell(self):
        # print(f'agent {self.id} sells')

        a_s = math.trunc(np.random.random()*self.A[-1])

        # Generate limit price
        s = self.market.p[-1]/np.random.normal(self.market.mu, self.market.sigma)

        self.s = s
        self.a_s = a_s