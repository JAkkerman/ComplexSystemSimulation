import numpy as np
import matplotlib.pyplot as plt


class Market():
    def __init__(self, p):
        self.p = [p] # stock price time series
        self.traders = []
        self.buyers = []
        self.sellers = []
        self.clusters = []
        self.T = 20
        self.k = 3.5
        self.mu = 1.01
        self.hist_vol = 0.1 # TODO: aanpassen aan historische vol
        self.sigma = self.update_sigma()

    def update_sigma(self):
        return self.k*self.hist_vol

    def reset_lists(self):
        self.buyers = []
        self.sellers = []
        self.traders = []

    def get_equilibrium_p(self):

        sorted_sell = sorted(self.sellers, key=lambda x: x.s_i)
        sorted_buy = sorted(self.buyers, key=lambda x: x.b_i)[::-1]

        # sorted_sell = sorted(MarketObj.sellers, key=lambda x: x.s_i)
        # sorted_buy = sorted(MarketObj.buyers, key=lambda x: x.b_i)[::-1]

        p_sell = [i.s_i for i in sorted_sell] # sorted list of sell price limits
        q_sell = np.cumsum([i.a_s for i in sorted_sell])
        p_buy = [i.b_i for i in sorted_buy] # sorted list of buy price limits
        q_buy = np.cumsum([i.a_b for i in sorted_buy])
        #print(q_sell)
        
        combined_buy = np.array([q_buy, p_buy])
        combined_sell = np.array([q_sell, p_sell])
        
        plt.figure()
        plt.scatter(combined_sell[0], combined_sell[1], label='Sell')
        plt.scatter(combined_buy[0],combined_buy[1], label='Buy')
        plt.ylabel('Price ($)')
        plt.xlabel('Cumulative Quantity of Stock')
        plt.legend(loc='best')
        plt.show()

        # print(len(sorted_sell))
        # print(len(sorted_buy))

        # Append zeroes such that both lists are of equal size
        if len(sorted_sell) > len(sorted_buy):
            sorted_buy += [0 for i in range(len(sorted_sell)-len(sorted_buy))]
        else:
            sorted_sell = [0 for i in range(len(sorted_buy)-len(sorted_sell))] + sorted_sell
            
        intersection = self.find_intersection(combined_buy, combined_sell)

        if len(intersection) == 0:
            return 0, [], []

        # print('Numpy array:', intersection)
        # print('Average clearing price:', np.mean(intersection[:,1]))
        # plt.show()

        buy_price_index = np.argmin(abs(np.array(p_buy) - np.mean(intersection[:,1])))
        buy_price = np.array(p_buy)[buy_price_index]
        buy_cum_quant = np.array(q_buy)[buy_price_index]
        # print('Buy Price:',buy_price)
        # print('Buy cum. quantity:',buy_cum_quant)
        
        
        sell_price_index = np.where((np.array(p_sell) - buy_price) < 0, 
                                        np.array(p_sell), -np.inf).argmax()
        # sell_price_index = np.argmin(abs(np.array(p_sell) - buy_price))
        sell_price = np.array(p_sell)[sell_price_index]
        sell_cum_quant = np.array(q_sell)[sell_price_index]
        # print('Sell Price:',sell_price)
        # print('Sell cum. quantity:',sell_cum_quant)

        transaction_q = min(sell_cum_quant, buy_cum_quant)
        self.p += [buy_price]

        if sell_cum_quant > buy_cum_quant:
            seller_index = np.where((q_sell - buy_cum_quant) < 0, 
                                    np.array(q_sell), -np.inf).argmax()

            # print('Last buyer q: ', sorted_buy[buy_price_index].a_b)
            # print('Last seller q: ', sorted_sell[seller_index].a_s)

            return transaction_q, sorted_sell[:seller_index+1], sorted_buy[:buy_price_index+1]

        else:
            buyer_index = np.where((q_buy - sell_cum_quant) > 0, 
                                    np.array(q_buy), np.inf).argmin()

            # print('Last buyer q: ', sorted_buy[buyer_index].a_b)
            # print('Last seller q: ', sorted_sell[sell_price_index].a_s)

            return transaction_q, sorted_sell[:sell_price_index+1], sorted_buy[:buyer_index+1]
        
        
    def perform_transactions(self, transaction_q, true_sellers, true_buyers):

        #print('sellers ', true_sellers)
        #print('buyers', true_buyers)

        # Perform buy transactions:
        sold_q = transaction_q
        for seller in true_sellers:
            if seller == 0:
                continue
            elif seller.a_s > sold_q:
                seller.C += [seller.C[-1] + seller.a_s*self.p[-1]]
                seller.A += [seller.A[-1] - seller.a_s]
                sold_q -= seller.a_s
            else:
                seller.C += [seller.C[-1] + sold_q*self.p[-1]]
                seller.A += [seller.A[-1] - sold_q]

        # Perfom sell transactions:
        bought_q = transaction_q
        for buyer in true_buyers:
            if buyer == 0:
                continue
            elif buyer.a_s > bought_q:
                buyer.C += [buyer.C[-1] + buyer.a_s*self.p[-1]]
                buyer.A += [buyer.A[-1] - buyer.a_s]
                bought_q -= buyer.a_s
            else:
                buyer.C += [buyer.C[-1] + bought_q*self.p[-1]]
                buyer.A += [buyer.A[-1] - bought_q]
        

    def find_intersection(self, combined_buy, combined_sell): # based on Bunkus' solution at https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value

        x1=list(combined_buy[0])
        y1=list(combined_buy[1])
        x2=list(combined_sell[0])
        y2=list(combined_sell[1])

        y_lists = y1[:]
        y_lists.extend(y2)
        y_dist = max(y_lists)/200.0
        
        x_lists = x1[:]
        x_lists.extend(x2)  
        x_dist = max(x_lists)/900.0
        division = 1000
        x_begin = min(x1[0], x2[0])     # 3
        x_end = max(x1[-1], x2[-1])     # 8
        
        points1 = [t for t in zip(x1, y1) if x_begin<=t[0]<=x_end]
        points2 = [t for t in zip(x2, y2) if x_begin<=t[0]<=x_end]
        # print points1
        # print points2
        
        x_axis = np.linspace(x_begin, x_end, division)
        #print('x_axis:',x_axis)
        idx = 0
        id_px1 = 0
        id_px2 = 0
        x1_line = []
        y1_line = []
        x2_line = []
        y2_line = []
        xpoints = len(x_axis)
        intersection = []
        while idx < xpoints:
            # Iterate over two line segments
            x = x_axis[idx]
            if id_px1>-1:
                if x >= points1[id_px1][0] and id_px1<len(points1)-1:
                    y1_line = np.linspace(points1[id_px1][1], points1[id_px1+1][1], 1000) # 1.4 1.401 1.402 etc. bis 2.1
                    x1_line = np.linspace(points1[id_px1][0], points1[id_px1+1][0], 1000)
                    id_px1 = id_px1 + 1
                    if id_px1 == len(points1):
                        x1_line = []
                        y1_line = []
                        id_px1 = -1
            if id_px2>-1:
                if x >= points2[id_px2][0] and id_px2<len(points2)-1:
                    y2_line = np.linspace(points2[id_px2][1], points2[id_px2+1][1], 1000)
                    x2_line = np.linspace(points2[id_px2][0], points2[id_px2+1][0], 1000)
                    id_px2 = id_px2 + 1
                    if id_px2 == len(points2):
                        x2_line = []
                        y2_line = []
                        id_px2 = -1
            if x1_line!=[] and y1_line!=[] and x2_line!=[] and y2_line!=[]:
                i = 0
                while abs(x-x1_line[i])>x_dist and i < len(x1_line)-1:
                    i = i + 1
                y1_current = y1_line[i]
                j = 0
                while abs(x-x2_line[j])>x_dist and j < len(x2_line)-1:
                    j = j + 1
                y2_current = y2_line[j]
                if abs(y2_current-y1_current)<y_dist and i != len(x1_line) and j != len(x2_line):
                    ymax = max(y1_current, y2_current)
                    ymin = min(y1_current, y2_current)
                    xmax = max(x1_line[i], x2_line[j])
                    xmin = min(x1_line[i], x2_line[j])
                    intersection.append((x, ymin+(ymax-ymin)/2))
                    plt.plot(x, y1_current, 'ro') # Plot the cross point
            idx += 1    
            # print("intersection points", intersection)

        return np.array(intersection)
        