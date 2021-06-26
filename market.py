import numpy as np
from scipy.optimize import fsolve
from arch import arch_model
import pandas as pd

# Our own defined modules
from cluster import Cluster
from data import calc_norm_return


class Market():
    def __init__(self, p, cluster, garch, garch_param, Pa, Pc):
        """
        Initialise a market.

        @param p                Initial price
        @param cluster          Whether to do clustering or not
        @param garch            Whether to determine volatility with GARCH or not
        @garch_param
        @param Pa               Probability of activating a cluster
        @param Pc               Probability of forming a pair between two agents, i.e clustering probability
        @param hist_vol         Initial volatility, default=0.1
        """
        self.p = [p]
        self.cluster=cluster
        self.traders = []
        self.buyers = []
        self.sellers = []
        self.clusters = []
        self.Pc = Pc
        self.Pa = Pa
        self.hist_vol = 0.1 # Value used by Raberto et al (2001)
        self.sigma = []
        self.pairs = []
        self.avg_degree = []

        self.garch = garch
        self.garch_param = garch_param
        self.garch_model = None

        self.update_sigma()

    def update_sigma(self):
        """
        Updates the volatility.
        """

        # Original volatility prediction, as specified in the paper of Raberto et al (2001)
        if not self.garch:
            self.sigma += [3.5 * self.hist_vol] # 3.5 is value used by Raberto et al (2001)

        # GARCH fitting
        else:
            # If lenght of price series is smaller than 20, use default value
            if len(self.p) < 20:
                self.sigma += [3.5 * self.hist_vol] # 3.5 is value used by Raberto et al (2001)
            # Otherwise, fit GARCH and predict one time step in the future
            else:
                self.sigma += [self.fit_GARCH()]


    def update_hist_vol(self):
        """
        Updates historical volatility by taking the standard deviation of the past price series.
        """

        # Only update if simulation has ran lang enough.
        if len(self.p) > 20:

            # Std of log-price returns of time window of 20 time steps
            returns = np.log(np.roll(np.array(self.p[:-20]), shift=-1) / np.array(self.p[:-20]))
            self.hist_vol = np.std(returns)

            # Update volatilty for this time-step
            self.update_sigma()


    def fit_GARCH(self):
        """
        Fits GARCH model to previous price data.
        """
        # Get last 20-100 data points and calculate the normalised returns
        price_data = pd.DataFrame(self.p[-min(len(self.p)-1, 100):])
        returns = 100 * calc_norm_return(price_data, False)

        # Make GARCH model
        am = arch_model(returns, p=self.garch_param[0], q=self.garch_param[1])
        res = am.fit(disp="off")
  
        # Perform volatility forecast
        forecasts = res.forecast(reindex=True)

        # Get the volaility forecast for tomorrow 
        return np.sqrt(forecasts.variance.iloc[-1][0])/100**2


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
        Activates one of the clusters with probability Pc, randomly selects activated cluster.
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
        for i in range(int(self.Pa*len(self.traders)**2)):
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

        # Sorted lists of sell/buy price limits
        p_sell = np.array([i.s_i for i in sorted_sell])
        p_buy = np.array([i.b_i for i in sorted_buy])

        # Total amounts of stock to buy/sell
        q_sell = np.cumsum([i.a_s for i in sorted_sell])
        q_buy = np.cumsum([i.a_b for i in sorted_buy])

        # Find the intersection of buy and sell curves
        intersection = self.find_intersection(p_buy, q_buy, p_sell, q_sell)

        if intersection == None:
            return 0, [], []

        # Find buyer closest to the intersection
        buy_price_index = np.where((np.array(p_buy) - intersection) > 0, np.array(p_buy), np.inf).argmin()
        buy_price = np.array(p_buy)[buy_price_index] # clearing price
        buy_cum_quant = np.array(q_buy)[buy_price_index]

        # Find seller closest to the intersection
        sell_price_index = np.where((np.array(p_sell) - buy_price) < 0, np.array(p_sell), -np.inf).argmax()
        sell_cum_quant = np.array(q_sell)[sell_price_index]

        # Determine transation quantity
        transaction_q = min(sell_cum_quant, buy_cum_quant)

        # Add new asset price
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
                # Seller can fill up his order
                if seller.a_s < sold_q:
                    seller.C += [seller.C[-1] + seller.a_s*self.p[-1]]
                    seller.A += [seller.A[-1] - seller.a_s]
                    sold_q -= seller.a_s
                # Partially fill up sell order
                else:
                    seller.C += [seller.C[-1] + sold_q*self.p[-1]]
                    seller.A += [seller.A[-1] - sold_q]
                    sold_q -= sold_q

        # Perform buy transactions:
        bought_q = transaction_q
        for buyer in true_buyers:
            if buyer != 0:
                # Buyer fills up his order
                if buyer.a_b < bought_q:
                    buyer.C += [buyer.C[-1] - buyer.a_b*self.p[-1]]
                    buyer.A += [buyer.A[-1] + buyer.a_b]
                    bought_q -= buyer.a_b
                # Partially fill buy order
                else:
                    buyer.C += [buyer.C[-1] - bought_q*self.p[-1]]
                    buyer.A += [buyer.A[-1] + bought_q]
                    bought_q -= bought_q

        # Keep asset and cash series consistent
        for trader in self.traders:
            if (trader not in true_sellers) and (trader not in true_buyers):
                trader.no_trade()

        # Empty buyer and seller lists
        self.reset_lists()


    def find_intersection(self, p_buy, q_buy, p_sell, q_sell):
        """
        Fits polynomial to buy and sell curves, finds intersection
        """

        # Fit the buy and sell curves
        buyfit = np.polyfit(q_buy[5:-5], p_buy[5:-5], deg=1)
        sellfit = np.polyfit(q_sell[5:-5], p_sell[5:-5], deg=1)

        # Make polynomial object
        buypol = np.poly1d(buyfit)
        sellpol = np.poly1d(sellfit)

        def solve_intersection(fun1, fun2, x0):
            return fsolve(lambda x : fun1(x) - fun2(x), x0)

        # Find the intersection between buy and sell 
        q_intersection = solve_intersection(buypol, sellpol, 100)
        p_intersection = buypol(q_intersection[0])

        # No intersection found
        if q_intersection[0] <= 0 or p_intersection <= 0:
            print('q: ', q_intersection[0], 'p: ', p_intersection)

            return None

        return p_intersection
