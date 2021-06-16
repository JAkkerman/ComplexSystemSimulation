import matplotlib.pyplot as plt
import numpy as np

def vis_market_cross(MarketObj):
    # buy_p = [buyer.b for buyer in MarketObj.buyers]
    # buy_q = [buyer.a_b for buyer in MarketObj.buyers]
    # sell_p = [seller.s for seller in MarketObj.sellers]
    # sell_q = [seller.a_s for seller in MarketObj.sellers]

    sorted_sell = sorted(MarketObj.sellers, key=lambda x: x.s_i)
    sorted_buy = sorted(MarketObj.buyers, key=lambda x: x.b_i)[::-1]

    p_sell = [i.s_i for i in sorted_sell] # sorted list of sell price limits
    q_sell = np.cumsum([i.a_s for i in sorted_sell])
    p_buy = [i.b_i for i in sorted_buy] # sorted list of buy price limits
    q_buy = np.cumsum([i.a_b for i in sorted_buy])
    
    #sets = [[p_sell, q_sell], [p_buy, q_buy]]
    #p_clearing = Intersection(sets)
    #print(f'Intersection: {p_clearing}')
    #pprint(vars(p_clearing))
    
    combined_buy = np.array([p_buy, q_buy])
    combined_sell = np.array([p_sell, q_sell])
    '''
    #print('Combined buy:',combined_buy)
    combined_buy = np.where(combined_buy[0][:] >= combined_sell[0][0], combined_buy[:], np.nan) # select relevant interval for buy curve
    #print('Combined buy adjusted:',combined_buy)
    combined_buy = combined_buy[:,~np.isnan(combined_buy).any(axis=0)] # drop NaN values
    print('Combined buy adjusted:',combined_buy)
    #print('Last relevant buy value:',combined_buy[0][0])
    #rint('Combined sell:',combined_sell)
    #print(combined_sell[0][0])
    combined_sell = np.where(combined_sell[0][:] <= combined_buy[0][0], combined_sell[:], np.nan) # select relevant interval for sell curve
    #print('Combined sell adjusted:',combined_sell)
    combined_sell = combined_sell[:,~np.isnan(combined_sell).any(axis=0)] # drop NaN values
    print('Combined sell adjusted:',combined_sell)
    print('Remaining values in sell curve:',len(combined_sell[0]))
    print('Remaining values in buy curve:',len(combined_buy[0]))
    
    min_list_size = min(len(combined_sell[0]), len(combined_buy[0]))
    
    difference_array = np.zeros(min_list_size)
    
    for i in range(min_list_size):
        difference_array[i] = combined_buy[1][min_list_size -1 - i] - combined_sell[1][i]
        if difference_array[i] < 0:
            print('Sign flip:',difference_array[i])
            print('Clearing price p* between',combined_buy[0][i],'and',combined_sell[0][i])
            break
    print('Difference array:',difference_array)
    '''
    plt.figure(dpi=450)
    plt.plot(q_sell, p_sell, label='Sell')
    plt.plot(q_buy, p_buy, label='Buy')
    plt.ylabel('Price ($)')
    plt.xlabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure(dpi=450)
    plt.scatter(combined_sell[1], combined_sell[0], label='Sell')
    plt.scatter(combined_buy[1],combined_buy[0], label='Buy')
    plt.ylabel('Price ($)')
    plt.xlabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.show()
    
    import sys
    
    fig = plt.figure(dpi=450)
    ax = fig.add_subplot(111)
    
    # x1 = [1,2,3,4,5,6,7,8]
    # y1 = [20,100,50,120,55,240,50,25]
    # x2 = [3,4,5,6,7,8,9]
    # y2 = [25,200,14,67,88,44,120]
    
    x1=list(combined_buy[1])
    y1=list(combined_buy[0])
    x2=list(combined_sell[1])
    y2=list(combined_sell[0])
    
    ax.plot(x1, y1, color='lightblue',linewidth=3, marker='s')
    ax.plot(x2, y2, color='darkgreen', marker='^')
    
    y_lists = y1[:]
    y_lists.extend(y2)
    y_dist = max(y_lists)/200.0
    
    x_lists = x1[:]
    x_lists.extend(x2)  
    x_dist = max(x_lists)/900.0
    division = 1000
    x_begin = min(x1[0], x2[0])     # 3
    x_end = max(x1[-1], x2[-1])     # 8
    
    points1 = [t for t in zip(x1, y1) if x_begin<=t[0]<=x_end]  # [(3, 50), (4, 120), (5, 55), (6, 240), (7, 50), (8, 25)]
    points2 = [t for t in zip(x2, y2) if x_begin<=t[0]<=x_end]  # [(3, 25), (4, 35), (5, 14), (6, 67), (7, 88), (8, 44)]
    # print points1
    # print points2
    
    x_axis = np.linspace(x_begin, x_end, division)
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
                ax.plot(x, y1_current, 'ro') # Plot the cross point
        idx += 1    
    print("intersection points", intersection)
    
    intersection = np.array(intersection)
    print('Numpy array:',intersection)
    print('Average clearing price:',np.mean(intersection[:,1]))
    plt.show()
    
    sell_price_index = np.argmin(abs(np.array(p_sell) - np.mean(intersection[:,1])))
    sell_price = np.array(p_sell)[sell_price_index]
    sell_cum_quant = np.array(q_sell)[sell_price_index]
    print('Sell Price:',sell_price)
    print('Sell cum. quantity:',sell_cum_quant)
    
    buy_price_index = np.argmin(abs(np.array(p_buy) - np.mean(intersection[:,1])))
    buy_price = np.array(p_buy)[buy_price_index]
    buy_cum_quant = np.array(q_buy)[buy_price_index]
    print('Buy Price:',buy_price)
    print('Buy cum. quantity:',buy_cum_quant)
    
    
    
    

