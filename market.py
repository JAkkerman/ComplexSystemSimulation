import numpy as np
from numpy.polynomial.chebyshev import chebfit
from scipy.optimize import fsolve

import matplotlib.pyplot as plt # TEMP

from cluster import Cluster


class Market():
    def __init__(self, p, cluster, T=20, k=3.5, mu=1.01, hist_vol=0.1, Pc=0.1, Pa=0.0002):
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
        self.sigma = self.update_sigma()
        self.pairs = []
        self.avg_degree = []

    def update_hist_vol(self):
        if len(self.p) > self.T:
            # print('T ', self.T)
            # print(self.p[:-self.T])
            # print(np.array(self.p[:-self.T]))
            returns = np.log(np.roll(np.array(self.p[:-self.T]), shift=-1)/np.array(self.p[:-self.T]))
            # print(np.std(returns))
            self.hist_vol = np.std(returns)
            self.update_sigma()

    def update_sigma(self):
        return self.k*self.hist_vol

    def reset_lists(self):
        self.buyers = []
        self.sellers = []

    def form_pairs(self):
        # pass
        # print(np.array())
        # print(set([[(i,j) for i in range(len(self.traders))] for j in range(len(self.traders))]))
        # return np.array([[(i,j) for i in range(len(self.traders)) if i!=j] for j in range(len(self.traders))])
        # return np.zeros(len(self.traders), len(self.traders))
        return []

    def init_cluster(self, members):
        ClusterObj = Cluster(members, self)
        self.clusters += [ClusterObj]
        for member in members:
            member.in_cluster = ClusterObj


    def activate_cluster(self):
        # print('jdhjkshfkjhsdhfjshdfskdjfhsdkjffhsd')
        if np.random.random() < self.Pc:
            # print('jdfsfjkshdjfhsdhfkshdfjshdhfhkskdfhksh')
            activated_cluster = np.random.choice(self.clusters)
            # print('yoot', activated_cluster)
            activated_cluster.activate()
            return activated_cluster
        # print('yoot')
        return None


    def merge_clusters(self, cluster1, cluster2):
        """
        Creates a new cluster with all members from the other clusters
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
        Makes decision on which clusters to form
        """

        # Randomly choose two individuals each time step and create cluster
        for i in range(2):
            pair  = np.random.choice(self.traders, size=2, replace=False)
            # print(pair)
            trader1 = pair[0]
            trader2 = pair[1]

            # Skip if already in the same cluster
            if trader1.in_cluster == trader2.in_cluster and trader1.in_cluster != None:
                continue
            # Add trader to cluster if other trader already in cluster
            elif trader1.in_cluster != None and trader2.in_cluster == None:
                # print('add2')
                # trader1.in_cluster.members += trader2
                trader1.in_cluster.add_to_cluster(trader2)
            elif trader1.in_cluster == None and trader2.in_cluster != None:
                # print('add1')
                trader2.in_cluster.add_to_cluster(trader1)

            # If both in different clusters, merge clusters
            elif trader1.in_cluster != None and trader2.in_cluster != None:
                # print('merge')
                self.merge_clusters(trader1.in_cluster, trader2.in_cluster)

            # If both in no cluster, make new cluster
            else:
                # print('init')
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

        combined_buy = np.array([p_buy, q_buy])
        combined_sell = np.array([p_sell, q_sell])

        # Append zeroes such that both lists are of equal size
        # if len(sorted_sell) > len(sorted_buy):
        #     sorted_buy += [0 for i in range(len(sorted_sell)-len(sorted_buy))]
        # else:
        #     sorted_sell = [0 for i in range(len(sorted_buy)-len(sorted_sell))] + sorted_sell

        intersection = self.find_intersection(combined_buy, combined_sell)

        if intersection == None:
            # print('yeet')
            return 0, [], []

        # if len(intersection) == 0:
        #     return 0, [], []

        # print('Numpy array:', intersection)
        # print('Average clearing price:', np.mean(intersection[:,1]))
        # plt.show()

        # buy_price_index = np.argmin(abs(np.array(p_buy) - np.mean(intersection[:,1])))
        buy_price_index = np.where((np.array(p_buy) - intersection) > 0,
                                np.array(p_buy), np.inf).argmin()
        # buy_price_index = np.argmin(abs(np.array(p_buy) - intersection))
        buy_price = np.array(p_buy)[buy_price_index]
        buy_cum_quant = np.array(q_buy)[buy_price_index]
        # print('Buy Price:',buy_price)
        # print('Buy cum. quantity:', buy_cum_quant)


        sell_price_index = np.where((np.array(p_sell) - buy_price) < 0,
                                        np.array(p_sell), -np.inf).argmax()
        # sell_price_index = np.argmin(abs(np.array(p_sell) - buy_price))
        # sell_price = np.array(p_sell)[sell_price_index]
        sell_cum_quant = np.array(q_sell)[sell_price_index]
        # print('Sell Price:',sell_price)
        # print('Sell cum. quantity:',sell_cum_quant)

        transaction_q = min(sell_cum_quant, buy_cum_quant)
        self.p += [buy_price]

        return transaction_q, sorted_sell, sorted_buy

        # print('q_buy', q_buy)

        # if sell_cum_quant > buy_cum_quant:
        #     seller_index = np.where((q_sell - buy_cum_quant) > 0,
        #                             np.array(q_sell), -np.inf).argmax()

        #     # print('seller index: ', seller_index)

        #     # print('Last buyer q: ', sorted_buy[buy_price_index].a_b)
        #     # print('Last seller q: ', sorted_sell[seller_index].a_s)

        #     return transaction_q, sorted_sell[:seller_index+1], sorted_buy[:buy_price_index+1]

        # else:
        #     buyer_index = np.where((q_buy - sell_cum_quant) < 0,
        #                             np.array(q_buy), np.inf).argmin()

        #     # print('buyer index: ', buyer_index)

        #     # print('Last buyer q: ', sorted_buy[buyer_index].a_b)
        #     # print('Last seller q: ', sorted_sell[sell_price_index].a_s)

        #     return transaction_q, sorted_sell[:sell_price_index+1], sorted_buy[:buyer_index+1]


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

        # if self.cluster:
        #     self.form_clusters()

        self.reset_lists()

    def find_intersection(self, combined_buy, combined_sell):

        # print(combined_sell[0])
        # print(combined_sell[1])

        buyfit = np.polyfit(combined_buy[1][5:-5], combined_buy[0][5:-5], deg=1)
        sellfit = np.polyfit(combined_sell[1][5:-5], combined_sell[0][5:-5], deg=1)

        buypol = np.poly1d(buyfit)
        sellpol = np.poly1d(sellfit)

        def solve_intersection(fun1, fun2, x0):
            return fsolve(lambda x : fun1(x) - fun2(x), x0)

        q_intersection = solve_intersection(buypol, sellpol, 100)
        p_intersection = buypol(q_intersection[0])

        # intersection = np.roots(buypol-sellpol)

        # print('q: ', q_intersection, 'p: ', p_intersection)

        # if len(self.p)%1000==0:
        #     q = np.arange(q_intersection+4000)
        #     plt.plot(q, buypol(q), label='buy', color='red')
        #     plt.plot(q, sellpol(q), label='sell', color='blue')
        #     plt.scatter(combined_buy[1], combined_buy[0], color='red')
        #     plt.scatter(combined_sell[1], combined_sell[0], color='blue')
        #     plt.scatter(q_intersection, p_intersection, color='black')
        #     plt.legend()
        #     plt.show()

        if q_intersection[0] <= 0:
            print('q: ', q_intersection[0], 'p: ', p_intersection)

            return None

        return p_intersection


    # def find_intersection(self, combined_buy, combined_sell):

        # x1=list(combined_buy[1])
        # y1=list(combined_buy[0])
        # x2=list(combined_sell[1])
        # y2=list(combined_sell[0])

        # y_lists = y1[:]
        # y_lists.extend(y2)
        # y_dist = max(y_lists)/200.0

        # x_lists = x1[:]
        # x_lists.extend(x2)
        # x_dist = max(x_lists)/900.0
        # division = 1000
        # x_begin = min(x1[0], x2[0])     # 3
        # x_end = max(x1[-1], x2[-1])     # 8

        # points1 = [t for t in zip(x1, y1) if x_begin<=t[0]<=x_end]  # [(3, 50), (4, 120), (5, 55), (6, 240), (7, 50), (8, 25)]
        # points2 = [t for t in zip(x2, y2) if x_begin<=t[0]<=x_end]  # [(3, 25), (4, 35), (5, 14), (6, 67), (7, 88), (8, 44)]
        # # print points1
        # # print points2

        # x_axis = np.linspace(x_begin, x_end, division)
        # idx = 0
        # id_px1 = 0
        # id_px2 = 0
        # x1_line = []
        # y1_line = []
        # x2_line = []
        # y2_line = []
        # xpoints = len(x_axis)
        # intersection = []
        # while idx < xpoints:
        #     # Iterate over two line segments
        #     x = x_axis[idx]
        #     if id_px1>-1:
        #         if x >= points1[id_px1][0] and id_px1<len(points1)-1:
        #             y1_line = np.linspace(points1[id_px1][1], points1[id_px1+1][1], 1000) # 1.4 1.401 1.402 etc. bis 2.1
        #             x1_line = np.linspace(points1[id_px1][0], points1[id_px1+1][0], 1000)
        #             id_px1 = id_px1 + 1
        #             if id_px1 == len(points1):
        #                 x1_line = []
        #                 y1_line = []
        #                 id_px1 = -1
        #     if id_px2>-1:
        #         if x >= points2[id_px2][0] and id_px2<len(points2)-1:
        #             y2_line = np.linspace(points2[id_px2][1], points2[id_px2+1][1], 1000)
        #             x2_line = np.linspace(points2[id_px2][0], points2[id_px2+1][0], 1000)
        #             id_px2 = id_px2 + 1
        #             if id_px2 == len(points2):
        #                 x2_line = []
        #                 y2_line = []
        #                 id_px2 = -1
        #     if x1_line!=[] and y1_line!=[] and x2_line!=[] and y2_line!=[]:
        #         i = 0
        #         while abs(x-x1_line[i])>x_dist and i < len(x1_line)-1:
        #             i = i + 1
        #         y1_current = y1_line[i]
        #         j = 0
        #         while abs(x-x2_line[j])>x_dist and j < len(x2_line)-1:
        #             j = j + 1
        #         y2_current = y2_line[j]
        #         if abs(y2_current-y1_current)<y_dist and i != len(x1_line) and j != len(x2_line):
        #             ymax = max(y1_current, y2_current)
        #             ymin = min(y1_current, y2_current)
        #             xmax = max(x1_line[i], x2_line[j])
        #             xmin = min(x1_line[i], x2_line[j])
        #             intersection.append((x, ymin+(ymax-ymin)/2))
        #             # ax.plot(x, y1_current, 'ro') # Plot the cross point
        #     idx += 1
        #     # print("intersection points", intersection)

        # return np.array(intersection)
