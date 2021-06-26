import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trader import Trader
from data import calc_norm_return, create_cdf, curve_fit_log, sample_gauss, SP500_pl, vol_cluster
import management

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


def vis_price_series(objects, N_time, N_agents, image_dir = None):

    # plt.plot(range(len(MarketObj.p)), MarketObj.p)
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.show()
    plt.figure(dpi=150)


        # popt, pcov, real = curve_fit_log(bins_count[i,1:], cdf[i])
        # if objects[i][1]:
        #     label = f"Herd model $\\alpha$ = {round(popt[1],2)}"
        # else:
        #     label = f"Model $\\alpha$ = {round(popt[1],2)}"
        # # plt.plot(bins_count[1:], real, label=label)
        # plt.scatter(bins_count[i,1:], cdf[i], label=label, marker='o')

    gaus_bins_count_list, gaus_cdf_list = [],[]
    for i in range(100):
        gaus_bins_count, gaus_cdf = sample_gauss(N_time)
        gaus_bins_count_list.append(gaus_bins_count)
        gaus_cdf_list.append(gaus_cdf)

    SP500_bins_count, SP500_cdf = SP500_pl()
    # gaus_bins_count, gaus_real = sample_gauss()

    fit_comparison_array_sp500= np.zeros((100*3, 5))
    j = 1
    sp500_model_array = np.array((SP500_cdf, SP500_bins_count[1:]))
    for i in range(1, int(100*2)):

        x_values = np.log10(np.delete(sp500_model_array[1], np.where(sp500_model_array[1] < j)))
        y_values = np.log10(np.delete(sp500_model_array[0], np.where(sp500_model_array[1] < j)))

        fit_comparison_array_sp500[i-1, 0] = x_values[0] # starting x_value for fit
        fit_comparison_array_sp500[i-1, 1] = x_values[-1] # final x_value for fit
        fit_comparison_array_sp500[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0,1]
        rsquared = correlation_xy**2
        fit_comparison_array_sp500[i-1, 4] = correlation_xy**2 # add R^2 value to array
        j = j + 0.01

    #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array_herd[i-1, 3]}')

    print(f'Slope for best fit SP500: {fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 3]}')
    # starting_x_sp500 = fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 0] # starting x herd
    # final_x_sp500 = fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 1] # final x herd
    # intercept_sp500 = fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 2] # intercept herd
    slope_sp500 = fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 3] # slope herd

    mean_cdf_gaus = np.mean(gaus_cdf_list, axis=0)
    mean_bin_gaus = np.mean(gaus_bins_count_list, axis=0)

    best_fit_array_gaus = np.zeros((100, 4))

    for k in range(100):

        fit_comparison_array_gaus= np.zeros((100*3, 5))
        j = 1
        gaus_model_array = np.array((gaus_cdf_list[k], gaus_bins_count_list[k][1:]))
        for i in range(1, int(100*2)):

            x_values = np.log10(np.delete(gaus_model_array[1], np.where(gaus_model_array[1] < j)))
            y_values = np.log10(np.delete(gaus_model_array[0], np.where(gaus_model_array[1] < j)))

            fit_comparison_array_gaus[i-1, 0] = x_values[0] # starting x_value for fit
            fit_comparison_array_gaus[i-1, 1] = x_values[-1] # final x_value for fit
            fit_comparison_array_gaus[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation_xy = correlation_matrix[0,1]
            rsquared = correlation_xy**2
            fit_comparison_array_gaus[i-1, 4] = correlation_xy**2 # add R^2 value to array
            j = j + 0.01

        #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array_herd[i-1, 3]}')

        print(f'Slope for best fit Gaus: {fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 3]}')
        best_fit_array_gaus[k][0] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 0] # starting x herd
        best_fit_array_gaus[k][1] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 1] # final x herd
        best_fit_array_gaus[k][2] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 2] # intercept herd
        best_fit_array_gaus[k][3] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 3] # slope herd

    mean_slope_gaus = np.mean(best_fit_array_gaus, axis=0)
    std_slope_gaus = np.std(best_fit_array_gaus, axis=0)
    confint_gaus = 1.96*(std_slope_gaus[3]/np.sqrt(100))

    for l in range(len(objects)):

        counter_herd = 0
        counter_norm = 0
        for object in objects[l]:
            if object[1][0]:
                counter_herd += 1
            else:
                counter_norm += 1

        cdf_herd = np.zeros((counter_herd, 50))
        cdf = np.zeros((counter_norm, 50))
        bins_count_herd = np.zeros((counter_herd, 51))
        bins_count = np.zeros((counter_norm, 51))

        counter_herd = 0
        counter_norm = 0

        label_list_herd = []
        label_list_norm = []
        N_agent_list_norm = []
        N_agent_list_herd = []

        for object in objects[l]:
            df = pd.DataFrame(object[0].p)
            df = calc_norm_return(df, True)
            #print(len(create_cdf(df)[0]))
            #cdf = np.zeros((2, len(create_cdf(df)[0])))
            #bins_count = np.zeros((2, len(create_cdf(df)[1])))

            # print(objects[i][1])

            if object[1][0]:

                cdf_herd[counter_herd,:], bins_count_herd[counter_herd,:] = create_cdf(df)
                counter_herd += 1
                N_agent_list_herd.append(object[1][1])
                # label_list_herd.append(f"Herd model, {object[1][1]} agents")
                label_list_herd.append(f"Herd model, Pc = {object[1][3]}")
                # label_list_herd.append(f"Herd model, Pa = {object[1][2]}")

            else:

                cdf[counter_norm,:], bins_count[counter_norm,:] = create_cdf(df)
                counter_norm += 1
                N_agent_list_norm.append(object[1][1])
                # label_list_norm.append(f"Model, {object[1][1]} agents, slope ")
                label_list_norm.append("Model")


        if cdf.shape[0] > 0:

            # model_array = np.array((cdf, bins_count))
            mean_cdf = np.mean(cdf, axis=0)
            mean_bin = np.mean(bins_count, axis=0)

            # Power law fits
            # Regular model
            best_fit_array = np.zeros((counter_norm, 4))
            for k in range(counter_norm):
                fit_comparison_array = np.zeros((N_agent_list_norm[0]*3, 5))
                j = 1
                model_array = np.array((cdf[k], bins_count[k][1:]))
                for i in range(1, int(N_agent_list_norm[0]*2)):
                    x_values = np.log10(np.delete(model_array[1], np.where(model_array[1] < j)))
                    y_values = np.log10(np.delete(model_array[0], np.where(model_array[1] < j)))
                    fit_comparison_array[i-1, 0] = x_values[0] # starting x_value for fit
                    fit_comparison_array[i-1, 1] = x_values[-1] # final x_value for fit
                    fit_comparison_array[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
                    correlation_matrix = np.corrcoef(x_values, y_values)
                    correlation_xy = correlation_matrix[0,1]
                    rsquared = correlation_xy**2
                    fit_comparison_array[i-1, 4] = correlation_xy**2 # add R^2 value to array
                    j = j + 0.01

                    #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array[i-1, 3]}')

                print(f'Slope for best fit regular model: {fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 3]}')
                best_fit_array[k][0] = fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 0] #starting x
                best_fit_array[k][1] = fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 1] # final x
                best_fit_array[k][2] = fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 2] # intercept
                best_fit_array[k][3] = fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 3] # slope

            mean_values_norm = np.mean(best_fit_array, axis=0)
            std_values_norm = np.std(best_fit_array, axis=0)
            confint_norm = 1.96*(std_values_norm[3]/np.sqrt(50))



            # plt.plot([10**mean_values_norm[0], 10**mean_values_norm[1]], [10**(mean_values_norm[2] + mean_values_norm[3]*mean_values_norm[0]), 10**(mean_values_norm[2] + mean_values_norm[3]*mean_values_norm[1])],
            #          label=f'Model fit, slope = {mean_values_norm[3]:.3f} $\\pm$ {std_values_norm[3]:.2f}', color='black') # plot regular model power law fit
            plt.scatter(mean_bin[1:], mean_cdf, label=label_list_norm[0] + f", slope = {round(mean_values_norm[3],2)} $\\pm$ {round(confint_norm,2)}")

        # Herd model
        if cdf_herd.shape[0] > 0:
            # print(cdf_herd)
            # print(bins_count_herd)
            # herd_model_array = np.array((cdf_herd, bins_count_herd))

            mean_cdf_herd = np.mean(cdf_herd, axis=0)
            mean_bin_herd = np.mean(bins_count_herd, axis=0)
            best_fit_herd_array = np.zeros((counter_herd, 4))
            for k in range(counter_herd):
                fit_comparison_array_herd = np.zeros((N_agent_list_herd[0]*3, 5))
                j = 1
                herd_model_array = np.array((cdf_herd[k], bins_count_herd[k][1:]))
                for i in range(1, int(N_agent_list_herd[0]*2)):

                    x_values = np.log10(np.delete(herd_model_array[1], np.where(herd_model_array[1] < j)))
                    y_values = np.log10(np.delete(herd_model_array[0], np.where(herd_model_array[1] < j)))
                    # x_values = np.log10(np.delete(bins_count_herd, np.where(bins_count_herd < j)))
                    # y_values = np.log10(np.delete(cdf_herd, np.where(bins_count_herd < j)))
                    fit_comparison_array_herd[i-1, 0] = x_values[0] # starting x_value for fit
                    fit_comparison_array_herd[i-1, 1] = x_values[-1] # final x_value for fit
                    fit_comparison_array_herd[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
                    correlation_matrix = np.corrcoef(x_values, y_values)
                    correlation_xy = correlation_matrix[0,1]
                    rsquared = correlation_xy**2
                    fit_comparison_array_herd[i-1, 4] = correlation_xy**2 # add R^2 value to array
                    j = j + 0.01

                #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array_herd[i-1, 3]}')

                print(f'Slope for best fit herd model: {fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 3]}')
                best_fit_herd_array[k][0] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 0] # starting x herd
                best_fit_herd_array[k][1] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 1] # final x herd
                best_fit_herd_array[k][2] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 2] # intercept herd
                best_fit_herd_array[k][3] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 3] # slope herd

            mean_values_herd = np.mean(best_fit_herd_array, axis=0)
            std_values_herd = np.std(best_fit_herd_array, axis=0)
            confint_herd = 1.96*(std_values_herd[3]/np.sqrt(50))

            # plt.plot([10**mean_values_herd[0], 10**mean_values_herd[1]], [10**(mean_values_herd[2] + mean_values_herd[3]*mean_values_herd[0]), 10**(mean_values_herd[2] + mean_values_herd[3]*mean_values_herd[1])],
            #          label=f'Herd model fit, slope = {mean_values_herd[3]:.3f} $\\pm$ {std_values_herd[3]:.2f}', color='black', linestyle='--') # plot herd model power law fit
            plt.scatter(mean_bin_herd[1:], mean_cdf_herd, label=label_list_herd[0] + f", slope = {round(mean_values_herd[3],2)} $\\pm$ {round(confint_herd,2)}")



    # Plotting
    # plt.scatter(gaus_bins_count[1:], gaus_cdf, label=f"Gaussian, slope = {round(slope_gaus, 2)}", marker='.')
    plt.scatter(mean_bin_gaus[1:], mean_cdf_gaus, label=f"Gaussian, slope = {round(mean_slope_gaus[3], 2)} $\\pm$ {round(confint_gaus, 2)}", marker='.')
    plt.scatter(SP500_bins_count[1:], SP500_cdf, label=f"S&P 500, slope = {round(slope_sp500,2)}", marker='.')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Normalized returns")
    plt.ylabel("Cumulative distribution")
    plt.title(f'{N_agents[0]} Traders, Pa = 0.0002, {N_time} Timesteps')
    # plt.title(f'{N_agents[0]} Traders, Pc = 0.1, {N_time} Timesteps')
    # plt.title(f'Pa = 0.0002, Pc = 0.1, {N_time} Timesteps')

    if image_dir != None:
        plt.savefig(image_dir)
    else:
        plt.show()


def vis_wealth_over_time(MarketObj, image_dir = None):

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,4))
    for TraderObj in MarketObj.traders:
        ax1.plot(range(len(TraderObj.C)), TraderObj.C, alpha=0.2)
    ax2.hist([TraderObj.C[-1] for TraderObj in MarketObj.traders])

    if image_dir != None:
        plt.savefig(image_dir)
    else:
        plt.show()

def cluster_vis(MarketObj, t, cluster, image_dir = None):
    if cluster:

        ret = calc_norm_return(pd.DataFrame(MarketObj.p), False)
        # max_val, min_val =  max(MarketObj.avg_degree),  min(MarketObj.avg_degree)
        mean_val, std_val =  np.mean(MarketObj.avg_degree),  np.std(MarketObj.avg_degree)
        # norm_degree = list(map(lambda x: (x - min_val)/(max_val - min_val), MarketObj.avg_degree))
        norm_degree = list(map(lambda x: (x - mean_val)/(std_val), MarketObj.avg_degree))

        plt.plot(np.linspace(0,len(ret),len(ret)), ret.values, color="blue", label="Stock returns", linewidth=0.5)
        plt.plot(np.linspace(0,t,t), norm_degree, color="orange", label="Avg network degree")
        plt.xlabel("Time")
        plt.xlim(450,850)
        # plt.ylabel("Normalized average network degree")
        plt.legend()
        plt.title("100 agents, Pa = 0.0002, Pc = 0.1")

        if image_dir != None:
            plt.savefig(image_dir)
        else:
            plt.show()

def vis_vol_cluster(objects, highp, window, N_time, image_dir = None):


    cluster_gaus, cluster_sp500 = vol_cluster(None, highp, window, N_time, True)
    count_gaus, bins_count_gaus = np.histogram(cluster_gaus, bins=[i for i in range(window+1)])
    count_sp500, bins_count_sp500 = np.histogram(cluster_sp500, bins=[i for i in range(window+1)])
    std_gaus = np.std(cluster_gaus)
    std_sp500 = np.std(cluster_sp500)


    for i in range(len(objects)):
        cluster_measures_norm = []
        cluster_measures_herd = []
        bin_count_herd = []
        count_herd = []
        bin_count_norm = []
        count_norm = []
        for object in objects[i]:
            df = pd.DataFrame(object[0].p)
            df = calc_norm_return(df, True)
            series = vol_cluster(df.values, highp, window, N_time, False)
            count_series, bins_count_series = np.histogram(series, bins=[i for i in range(window+1)])
            std_series = np.std(series)
            cluster_measure = std_series/std_gaus
            if object[1][0]:
                cluster_measures_herd.append(cluster_measure)
                bin_count_herd.append(bins_count_series[1:])
                count_herd.append(count_series)
                # label = f"Herd model, R = {round(cluster_measure,2)}"
            else:
                cluster_measures_norm.append(cluster_measure)
                bin_count_norm.append(bins_count_series[1:])
                count_norm.append(count_series)
                # label = f"Model, R = {round(cluster_measure,2)}"
            # plt.plot(bins_count_series[1:], count_series, label=label)

        if len(count_norm) > 0:

            mean_count_norm = np.mean(np.array(count_norm), axis=0)
            mean_measure_norm = np.mean(cluster_measures_norm)
            std_measure_norm = np.std(cluster_measures_norm)
            conf_norm = 1.96*(std_measure_norm/np.sqrt(len(count_norm)))
            plt.plot(bin_count_norm[0], mean_count_norm, label=f"Model, R = {round(mean_measure_norm,2)} $\\pm$ {round(std_measure_norm, 2)}")
            plt.plot(bin_count_norm[0], mean_count_norm, label=f"Model, {object[1][1]} agents, R = {round(mean_measure_norm,2)} $\\pm$ {round(conf_norm, 2)}")

        if len(count_herd) > 0:

            mean_count_herd = np.mean(np.array(count_herd), axis=0)
            mean_measure_herd = np.mean(cluster_measures_herd)
            std_measure_herd = np.std(cluster_measures_herd)
            conf_herd = 1.96*(std_measure_herd/np.sqrt(len(count_herd)))
            plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, Pc = {object[1][3]}, R = {round(mean_measure_herd,2)} $\\pm$ {round(conf_herd, 2)}")
            # plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, {object[1][1]} agents, R = {round(mean_measure_herd,2)} $\\pm$ {round(std_measure_herd, 2)}")
            # plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, Pa = {object[1][2]}, R = {round(mean_measure_herd,2)} $\\pm$ {round(std_measure_herd, 2)}")



    plt.plot(bins_count_sp500[1:], count_sp500, label=f"SP500, R = {round(std_sp500/std_gaus, 2)}")
    plt.plot(bins_count_gaus[1:], count_gaus, label="Gaussian distribution")
    plt.xlabel("Number of trading days")
    plt.ylabel("Frequency")
    plt.title("Volatility clustering, 100 Agents, Pa = 0.0002")
    # plt.title("Volatility clustering, Pa = 0.0002, Pc = 0.1")
    # plt.title("Volatility clustering, 100 Agents, Pc = 0.1")
    plt.yscale("log")
    plt.legend()
    if image_dir != None:
        plt.savefig(image_dir)
    else:
        plt.show()


def plot_lorenz_curve(MarketObj, image_dir=None):
    """
    Plots the Lorenz curve
    """
    all_t = [1000, 5000, 10000]

    fig = plt.figure(figsize=(6,6))

    for t in all_t:
        print(len(MarketObj.p))

        sorted_wealth = sorted(MarketObj.traders, key=lambda x: x.A[t-1]*MarketObj.p[t-1] + x.C[t-1])
        sorted_wealth = [i.A[t-1]*MarketObj.p[t-1] + i.C[t-1] for i in sorted_wealth]
        cum_wealth = np.cumsum(sorted_wealth)

        X = np.linspace(0, 1, len(MarketObj.traders))
        G = np.abs(1 - sum([(X[i+1]-X[i])*(cum_wealth[i+1]/sum(sorted_wealth)+cum_wealth[i]/sum(sorted_wealth))
                            for i in range(len(MarketObj.traders)-1)]))

        plt.plot(np.linspace(0,1,100), cum_wealth/sum(sorted_wealth), label=f't={t}, Gini={round(G,2)}')

    plt.plot([0,1], [0,1], linestyle='--', color='black')
    plt.title(f'Lorenz curve')
    plt.xlabel('Cumulative share of agents')
    plt.ylabel('Cumulative share of income')
    plt.legend()
    if image_dir != None:
        plt.savefig(image_dir)
    else:
        plt.show()


def vis_volatility_series(objects, N_time):
    x = np.linspace(0, len(objects[0][0].sigma), len(objects[0][0].sigma))
    for object in objects:
        vol = object[0].sigma
        plt.plot(x, vol)
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.show()

def visualiseSingleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i):

    MarketObj = management.loadSingleMarket(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i)
    print(f'Loaded object, now start visualising for {i}')

    # Image directory
    image_dir = 'images/Nagents{N_agents}_Pa{Pa}_Pc{Pc}_i{i}'

    # Do all possible visualisations for a single market
    # vis_wealth_over_time(MarketObj, image_dir)
    #vis_price_series([MarketObj], N_time, N_agents, image_dir)
    cluster_vis(MarketObj, N_time, cluster, image_dir)
    # plot_lorenz_curve(MarketObj, image_dir)

def visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N):
    object_list = []
    for a in Pa:
        for c in Pc:
            for agents in N_agents:
                for cluster in [True, False]:
                    object_list.append(management.loadMultipleMarkets(agents, N_time, C, A, p, garch, garch_n, garch_param, a, c, cluster, N))
    print("Loaded objects, now start visualising")

    # Image directory
    image_dir = 'images/Nagents{N_agents}_Pa{Pa}_Pc{Pc}'

    # vis_price_series(object_list, N_time, N_agents, image_dir)
    vis_vol_cluster(object_list, 0.2, 10, N_time, image_dir)
