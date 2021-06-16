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
    
    plt.figure(dpi=450)
    plt.plot(p_sell, q_sell, label='Sell')
    plt.plot(p_buy, q_buy, label='Buy')
    plt.xlabel('Price ($)')
    plt.ylabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure(dpi=450)
    plt.scatter(combined_sell[0], combined_sell[1], label='Sell')
    plt.scatter(combined_buy[0],combined_buy[1], label='Buy')
    plt.xlabel('Price ($)')
    plt.ylabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.show()

