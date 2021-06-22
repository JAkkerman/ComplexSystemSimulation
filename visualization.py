import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trader import Trader
from data import calc_norm_return, create_cdf, curve_fit_log, sample_gauss, SP500_pl, vol_cluster

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
    # plt.figure(dpi=450)
    # plt.plot(q_sell, p_sell, label='Sell')
    # plt.plot(q_buy, p_buy, label='Buy')
    # plt.ylabel('Price ($)')
    # plt.xlabel('Cumulative Quantity of Stock')
    # plt.legend(loc='best')
    # plt.show()

    plt.figure()
    plt.scatter(combined_sell[1], combined_sell[0], label='Sell')
    plt.scatter(combined_buy[1],combined_buy[0], label='Buy')
    plt.ylabel('Price ($)')
    plt.xlabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.show()

    import sys

    fig = plt.figure()
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


def vis_price_series(objects, N_time):

    # plt.plot(range(len(MarketObj.p)), MarketObj.p)
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.show()

    for i in range(len(objects)):

        df = pd.DataFrame(objects[i][0].p)
        df = calc_norm_return(df, True)
        cdf, bins_count = create_cdf(df)
        popt, pcov, real = curve_fit_log(bins_count[1:], cdf)
        if objects[i][1]:
            label = f"Herd model $\\alpha$ = {round(popt[1],2)}"
        else:
            label = f"Model $\\alpha$ = {round(popt[1],2)}"
        # plt.plot(bins_count[1:], real, label=label)
        plt.plot(bins_count[1:], cdf, label=label)

    gaus_bins_count, gaus_cdf = sample_gauss(N_time)
    SP500_bins_count, SP500_cdf = SP500_pl()
    # gaus_bins_count, gaus_real = sample_gauss()

    plt.plot(gaus_bins_count[1:], gaus_cdf, label="Gaussian distribution")
    plt.plot(SP500_bins_count[1:], SP500_cdf, label="SP500")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Normalized returns")
    plt.ylabel("Cumulative distribution")
    plt.show()


def vis_wealth_over_time(MarketObj):

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,4))
    for TraderObj in MarketObj.traders:
        ax1.plot(range(len(TraderObj.C)), TraderObj.C, alpha=0.2)
    ax2.hist([TraderObj.C[-1] for TraderObj in MarketObj.traders])

    plt.show()

def cluster_vis(MarketObj, t, cluster):
    if cluster:

        ret = calc_norm_return(pd.DataFrame(MarketObj.p), False)
        # max_val, min_val =  max(MarketObj.avg_degree),  min(MarketObj.avg_degree)
        mean_val, std_val =  np.mean(MarketObj.avg_degree),  np.std(MarketObj.avg_degree)
        # norm_degree = list(map(lambda x: (x - min_val)/(max_val - min_val), MarketObj.avg_degree))
        norm_degree = list(map(lambda x: (x - mean_val)/(std_val), MarketObj.avg_degree))

        plt.plot(np.linspace(0,t,t), ret.values, color="blue", label="Stock returns", linewidth=0.5)
        plt.plot(np.linspace(0,t,t), norm_degree, color="orange", label="Avg network degree")
        plt.xlabel("Time")
        # plt.ylabel("Normalized average network degree")
        plt.legend()
        plt.show()

def vis_vol_cluster(objects, highp, window, N_time):

    cluster_gaus, cluster_sp500 = vol_cluster(None, highp, window, N_time, True)
    count_gaus, bins_count_gaus = np.histogram(cluster_gaus, bins=[i for i in range(window+1)])
    count_sp500, bins_count_sp500 = np.histogram(cluster_sp500, bins=[i for i in range(window+1)])
    std_gaus = np.std(cluster_gaus)
    std_sp500 = np.std(cluster_sp500)

    for object in objects:
        df = pd.DataFrame(object[0].p)
        df = calc_norm_return(df, True)
        series = vol_cluster(df.values, highp, window, N_time, False)
        count_series, bins_count_series = np.histogram(series, bins=[i for i in range(window+1)])
        std_series = np.std(series)
        cluster_measure = std_series/std_gaus
        if object[1]:
            label = f"Herd model, R = {round(cluster_measure,2)}"
        else:
            label = f"Model, R = {round(cluster_measure,2)}"
        plt.plot(bins_count_series[1:], count_series, label=label)


    plt.plot(bins_count_gaus[1:], count_gaus, label="Gaussian distribution")
    plt.plot(bins_count_sp500[1:], count_sp500, label=f"SP500, R = {round(std_sp500/std_gaus, 2)}")
    plt.xlabel("Number of trading days")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend()
    plt.show()
