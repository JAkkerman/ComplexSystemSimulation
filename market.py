import numpy as np
from numpy.polynomial.chebyshev import chebfit
from scipy.optimize import fsolve

from arch import arch_model
import datetime as dt
import arch.data.sp500
import pandas as pd

import matplotlib.pyplot as plt # TEMP

from cluster import Cluster


class Market():
    def __init__(self, p, cluster, garch, garch_param,
                 T=20, k=3.5, mu=1.01, hist_vol=0.1, Pc=0.1, Pa=0.0002):
        self.p = [p]
        self.cluster=cluster
        self.traders = []
        self.buyers = []
        self.sellers = []
        self.clusters = []
        self.T = T
        self.k = k
        self.mu = mu
        self.Pc = Pc
        self.Pa = Pa
        self.hist_vol = hist_vol # TODO: aanpassen aan historische vol
        self.sigma = []
        self.pairs = []
        self.avg_degree = []

        self.garch = garch
        self.garch_param = garch_param
        self.garch_model = None

        self.update_sigma()


    def update_sigma(self):
        """
        Updates sigma.
        """
        # Original volatility prediction, as specified in Raberto et al (2001)
        if not self.garch:
            self.sigma += [self.k*self.hist_vol]
        
        # GARCH fitting, if selected
        else:
            # If lenght of price series is smaller than 20, use default value
            if len(self.p) < 100:
                self.sigma += [self.k*self.hist_vol]
            # Otherwise, fit GARCH and predict one time step in the future
            else:
                self.sigma += [self.fit_GARCH()]


    def update_hist_vol(self):
        """
        Updates historical volatility by taking the standard deviation of the 
            past price series.
        """
        if len(self.p) > self.T:
            returns = np.log(np.roll(np.array(self.p[:-self.T]), shift=-1)/np.array(self.p[:-self.T]))
            self.hist_vol = np.std(returns)
            self.update_sigma()


    def fit_GARCH(self):
        """
        Fits GARCH model to previous price data.
        """
        # Get last 20-100 data points
        price_data = pd.DataFrame(self.p[-min(len(self.p)-1, 100):])
        returns = 100*price_data.pct_change().dropna()

        am = arch_model(returns, p=self.garch_param[0], q=self.garch_param[1])
        res = am.fit()
        forecasts = res.forecast(reindex=True)

        print(forecasts.variance)

        print(np.sqrt(forecasts.variance.iloc[-1][0]))
        return np.sqrt(forecasts.variance.iloc[-1][0])


    def reset_lists(self):
        """
        Resets lists, deletes buyers and sellers from this iteration.
        """
        self.buyers = []
        self.sellers = []


    def init_cluster(self, members):
        """
        Initializes cluster object
        """
        ClusterObj = Cluster(members, self)
        self.clusters += [ClusterObj]
        for member in members:
            member.in_cluster = ClusterObj


    def activate_cluster(self):
        """
        Activates one of the clusters with probability Pc, 
            randomly selects activated cluster.
        """
        if np.random.random() < self.Pc:
            activated_cluster = np.random.choice(self.clusters)
            activated_cluster.activate()
            return activated_cluster
        return None


    def merge_clusters(self, cluster1, cluster2):
        """
        Creates a new cluster with all members from the other clusters.
        """
        all_members = cluster1.members + cluster2.members
        merged_cluster = Cluster(all_members, self)
        for member in all_members:
            member.in_cluster = merged_cluster

        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.clusters += [merged_cluster]


    def form_clusters(self):
        """
        Makes decision on which clusters to form.
        """

        # Randomly choose two individuals each time step and create cluster
        for i in range(2):
            pair  = np.random.choice(self.traders, size=2, replace=False)
            trader1 = pair[0]
            trader2 = pair[1]

            # Skip if already in the same cluster
            if trader1.in_cluster == trader2.in_cluster and trader1.in_cluster != None:
                continue
            # Add trader to cluster if other trader already in cluster
            elif trader1.in_cluster != None and trader2.in_cluster == None:
                trader1.in_cluster.add_to_cluster(trader2)
            elif trader1.in_cluster == None and trader2.in_cluster != None:
                trader2.in_cluster.add_to_cluster(trader1)

            # If both in different clusters, merge clusters
            elif trader1.in_cluster != None and trader2.in_cluster != None:
                self.merge_clusters(trader1.in_cluster, trader2.in_cluster)

            # If both in no cluster, make new cluster
            else:
                self.init_cluster([trader1, trader2])


    def get_equilibrium_p(self):
        """
        Determines clearing price and quantity
        """

        # Sort buyers and sellers based on their limit prices
        sorted_sell = sorted(self.sellers, key=lambda x: x.s_i)
        sorted_buy = sorted(self.buyers, key=lambda x: x.b_i, reverse=True)

        p_sell = [i.s_i for i in sorted_sell] # sorted list of sell price limits
        q_sell = np.cumsum([i.a_s for i in sorted_sell])
        p_buy = [i.b_i for i in sorted_buy] # sorted list of buy price limits
        q_buy = np.cumsum([i.a_b for i in sorted_buy])

        intersection = self.find_intersection(p_buy, q_buy, p_sell, q_sell)

        if intersection == None:
            return 0, [], []

        # Find buyer closest to the intersection
        buy_price_index = np.where((np.array(p_buy) - intersection) > 0,
                                np.array(p_buy), np.inf).argmin()
        buy_price = np.array(p_buy)[buy_price_index]
        buy_cum_quant = np.array(q_buy)[buy_price_index]

        # Find seller closest to the intersection
        sell_price_index = np.where((np.array(p_sell) - buy_price) < 0,
                                        np.array(p_sell), -np.inf).argmax()
        sell_cum_quant = np.array(q_sell)[sell_price_index]

        # Determine transation quantity
        transaction_q = min(sell_cum_quant, buy_cum_quant)
        self.p += [buy_price]

        return transaction_q, sorted_sell, sorted_buy


    def perform_transactions(self, transaction_q, true_sellers, true_buyers):
        """
        Performs buy and sell transactions,
            changes asset and cash balance of true buyers and sellers
        """

        # Perform sell transactions:
        sold_q = transaction_q
        for seller in true_sellers:
            if seller != 0:
                if seller.a_s < sold_q:
                    seller.C += [seller.C[-1] + seller.a_s*self.p[-1]]
                    seller.A += [seller.A[-1] - seller.a_s]
                    sold_q -= seller.a_s
                else:
                    seller.C += [seller.C[-1] + sold_q*self.p[-1]]
                    seller.A += [seller.A[-1] - sold_q]
                    sold_q -= sold_q

        # Perform buy transactions:
        bought_q = transaction_q
        for buyer in true_buyers:
            if buyer != 0:
                if buyer.a_s < bought_q:
                    buyer.C += [buyer.C[-1] - buyer.a_b*self.p[-1]]
                    buyer.A += [buyer.A[-1] + buyer.a_b]
                    bought_q -= buyer.a_b
                else:
                    buyer.C += [buyer.C[-1] - bought_q*self.p[-1]]
                    buyer.A += [buyer.A[-1] + bought_q]
                    bought_q -= bought_q

        for trader in self.traders:
            if (trader not in true_sellers) and (trader not in true_buyers):
                trader.no_trade()

        self.reset_lists()


    def find_intersection(self, p_buy, q_buy, p_sell, q_sell):
        """
        Fits polynomial to buy and sell curves, finds intersection
        """

        buyfit = np.polyfit(q_buy[5:-5], p_buy[5:-5], deg=1)
        sellfit = np.polyfit(q_sell[5:-5], p_sell[5:-5], deg=1)

        buypol = np.poly1d(buyfit)
        sellpol = np.poly1d(sellfit)

        def solve_intersection(fun1, fun2, x0):
            return fsolve(lambda x : fun1(x) - fun2(x), x0)

        q_intersection = solve_intersection(buypol, sellpol, 100)
        p_intersection = buypol(q_intersection[0])

        if q_intersection[0] <= 0:
            print('q: ', q_intersection[0], 'p: ', p_intersection)

            return None

        return p_intersection