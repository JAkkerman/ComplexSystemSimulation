import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import management
import os

# Our own defined modules
from trader import Trader
from data import calc_norm_return, create_cdf, curve_fit_log, sample_gauss, SP500_pl, vol_cluster


def vis_market_cross(MarketObj, transaction_q):
    # buy_p = [buyer.b for buyer in MarketObj.buyers]
    # buy_q = [buyer.a_b for buyer in MarketObj.buyers]
    # sell_p = [seller.s for seller in MarketObj.sellers]
    # sell_q = [seller.a_s for seller in MarketObj.sellers]

    sorted_sell = sorted(MarketObj.sellers, key=lambda x: x.s_i)[10:-10]
    sorted_buy = sorted(MarketObj.buyers, key=lambda x: x.b_i)[::-1][10:-10]

    print(sorted_sell)
    print(sorted_buy)

    p_sell = [i.s_i for i in sorted_sell] # sorted list of sell price limits
    q_sell = np.cumsum([i.a_s for i in sorted_sell])
    p_buy = [i.b_i for i in sorted_buy] # sorted list of buy price limits
    q_buy = np.cumsum([i.a_b for i in sorted_buy])

    demand = np.polyfit(q_buy, p_buy, deg=1)
    supply = np.polyfit(q_sell, p_sell, deg=1)

    demand = np.poly1d(demand)
    supply = np.poly1d(supply)

    #sets = [[p_sell, q_sell], [p_buy, q_buy]]
    # p_clearing = Intersection(sets)
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

    X = np.arange(np.average(q_sell) - 2000, np.average(q_sell) + 2000)

    plt.figure()
    plt.scatter(q_sell, p_sell, label='Sell')
    plt.plot(X, supply(X))
    plt.scatter(q_buy, p_buy, label='Buy')
    plt.plot(X, demand(X))
    # plt.scatter(transaction_q, MarketObj.p[-1], color='red', alpha=0.2)
    plt.ylabel('Price ($)')
    plt.xlabel('Cumulative Quantity of Stock')
    plt.legend(loc='best')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.show()



def vis_price_series(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, N):

    plt.figure(dpi=150)

    #####################
    #  Start plotting and fitting gaussian and S&P 500 
    gaus_bins_count_list, gaus_cdf_list = [], []
    for i in range(100):
        gaus_bins_count, gaus_cdf = sample_gauss(N_time)
        gaus_bins_count_list.append(gaus_bins_count)
        gaus_cdf_list.append(gaus_cdf)

    SP500_bins_count, SP500_cdf = SP500_pl()

    fit_comparison_array_sp500 = np.zeros((100*3, 5))
    j = 1
    sp500_model_array = np.array((SP500_cdf, SP500_bins_count[1:]))
    for i in range(1, int(100*2)):

        x_values = np.log10(np.delete(sp500_model_array[1], np.where(sp500_model_array[1] < j)))
        y_values = np.log10(np.delete(sp500_model_array[0], np.where(sp500_model_array[1] < j)))

        fit_comparison_array_sp500[i-1, 0] = x_values[0] # starting x_value for fit
        fit_comparison_array_sp500[i-1, 1] = x_values[-1] # final x_value for fit
        fit_comparison_array_sp500[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0, 1]
        rsquared = correlation_xy**2
        fit_comparison_array_sp500[i-1, 4] = correlation_xy**2 # add R^2 value to array
        j = j + 0.01

    #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array_herd[i-1, 3]}')

    print(f'Slope for best fit SP500: {fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:,4]), 3]}')
    slope_sp500 = fit_comparison_array_sp500[np.argmax(fit_comparison_array_sp500[:, 4]), 3] # slope herd

    mean_cdf_gaus = np.mean(gaus_cdf_list, axis=0)
    mean_bin_gaus = np.mean(gaus_bins_count_list, axis=0)

    best_fit_array_gaus = np.zeros((100, 4))

    for k in range(100):

        fit_comparison_array_gaus = np.zeros((100*3, 5))
        j = 1
        gaus_model_array = np.array((gaus_cdf_list[k], gaus_bins_count_list[k][1:]))
        for i in range(1, int(100*2)):

            x_values = np.log10(np.delete(gaus_model_array[1], np.where(gaus_model_array[1] < j)))
            y_values = np.log10(np.delete(gaus_model_array[0], np.where(gaus_model_array[1] < j)))

            fit_comparison_array_gaus[i-1, 0] = x_values[0] # starting x_value for fit
            fit_comparison_array_gaus[i-1, 1] = x_values[-1] # final x_value for fit
            fit_comparison_array_gaus[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1) # fit line
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation_xy = correlation_matrix[0, 1]
            rsquared = correlation_xy**2
            fit_comparison_array_gaus[i-1, 4] = correlation_xy**2 # add R^2 value to array
            j = j + 0.01

        #print(f'Normalized returns >= {x_values[0]}, R-squared value: {rsquared}, power law fit slope: {fit_comparison_array_herd[i-1, 3]}')

        print(f'Slope for best fit Gaus: {fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:,4]), 3]}')
        best_fit_array_gaus[k][0] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:, 4]), 0] # starting x herd
        best_fit_array_gaus[k][1] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:, 4]), 1] # final x herd
        best_fit_array_gaus[k][2] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:, 4]), 2] # intercept herd
        best_fit_array_gaus[k][3] = fit_comparison_array_gaus[np.argmax(fit_comparison_array_gaus[:, 4]), 3] # slope herd

    mean_slope_gaus = np.mean(best_fit_array_gaus, axis=0)
    std_slope_gaus = np.std(best_fit_array_gaus, axis=0)
    confint_gaus = 1.96*(std_slope_gaus[3]/np.sqrt(100))

    #### End plotting and fitting gaussian and S&P 500 
    #######################################

    standard_Pa = 0.0002
    standard_Pc = 0.1
    standard_Nagent = 100
    is_Pa_Experiment, is_Pc_experiment, is_Nagent_experiment = False, False, False

    # Perform visualisation for every pair
    for pair in itertools.combinations([standard_Pa, standard_Pc, standard_Nagent], 2):

        # Get the variable we want to vary
        is_Pa_Experiment = standard_Pa not in pair
        is_Pc_experiment = standard_Pc not in pair
        is_Nagent_experiment = standard_Nagent not in pair

        # Pa experiment
        if is_Pa_Experiment:
            Pc = [standard_Pc]
            agents = [standard_Nagent]
            Pa = Pa_list
            print("PA")
        # Pc experiment    
        elif is_Pc_experiment:
            Pa = [standard_Pa]
            agents = [standard_Nagent]
            Pc = Pc_list
            print("PC")
        # Agent experiment
        elif is_Nagent_experiment:
            Pc = [standard_Pc]
            Pa = [standard_Pa]
            agents = N_agents
            print("agent")

        # Image directory
        image_dir = f'images/Nagents{agents}_Pa{Pa}_Pc{Pc}/'
        # Make directory
        if not(os.path.isdir(image_dir)):
            os.mkdir(image_dir)

        # Loop over all configurations (in reality, 2 out of 3 are fixed and 1 real loop is performed)
        for a in Pa:
            for c in Pc:
                for agent in agents:
                    print(a)
                    print(c)
                    print(agent)
                    # Load all markets for this configuration
                    objects = management.loadMultipleMarkets(agent, N_time, C, A, p, garch, garch_n, garch_param, a, c, cluster, N)

                    counter_herd = 0
                    counter_norm = 0
                    for object in objects:
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

                    for object in objects:
                        df = pd.DataFrame(object[0].p)
                        df = calc_norm_return(df, True)

                        if object[1][0]:
                            cdf_herd[counter_herd, :], bins_count_herd[counter_herd, :] = create_cdf(df)
                            counter_herd += 1
                            N_agent_list_herd.append(object[1][1])
                        else:
                            cdf[counter_norm,:], bins_count[counter_norm,:] = create_cdf(df)
                            counter_norm += 1
                            N_agent_list_norm.append(object[1][1])
                            label_list_norm.append(f"Model, {object[1][1]} agents, slope ")
                            label_list_norm.append("Model")

                        # Plot label
                        if is_Pa_Experiment:
                            label_list_herd.append(f"Herd model, Pa = {a}")
                        elif is_Pc_experiment:
                            label_list_herd.append(f"Herd model, Pc = {c}")
                        elif is_Nagent_experiment:
                            label_list_herd.append(f"Herd model, {agent} agents")


                    if cdf.shape[0] > 0:

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
                        confint_norm = 1.96*(std_values_norm[3]/np.sqrt(15))

                        # plt.plot([10**mean_values_norm[0], 10**mean_values_norm[1]], [10**(mean_values_norm[2] + mean_values_norm[3]*mean_values_norm[0]), 10**(mean_values_norm[2] + mean_values_norm[3]*mean_values_norm[1])],
                        #          label=f'Model fit, slope = {mean_values_norm[3]:.3f} $\\pm$ {std_values_norm[3]:.2f}', color='black') # plot regular model power law fit
                        plt.scatter(mean_bin[1:], mean_cdf, label=label_list_norm[0] + f", slope = {round(mean_values_norm[3],2)} $\\pm$ {round(confint_norm,2)}")

                    # Herd model
                    if cdf_herd.shape[0] > 0:
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
                                fit_comparison_array_herd[i-1, 0] = x_values[0] # starting x_value for fit
                                fit_comparison_array_herd[i-1, 1] = x_values[-1] # final x_value for fit
                                fit_comparison_array_herd[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1)  # fit line
                                correlation_matrix = np.corrcoef(x_values, y_values)
                                correlation_xy = correlation_matrix[0, 1]
                                rsquared = correlation_xy**2
                                fit_comparison_array_herd[i-1, 4] = correlation_xy**2 # add R^2 value to array
                                j = j + 0.01

                            print(f'Slope for best fit herd model: {fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:,4]), 3]}')
                            best_fit_herd_array[k][0] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:, 4]), 0]  # starting x herd
                            best_fit_herd_array[k][1] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:, 4]), 1]  # final x herd
                            best_fit_herd_array[k][2] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:, 4]), 2]  # intercept herd
                            best_fit_herd_array[k][3] = fit_comparison_array_herd[np.argmax(fit_comparison_array_herd[:, 4]), 3]  # slope herd

                        mean_values_herd = np.mean(best_fit_herd_array, axis=0)
                        std_values_herd = np.std(best_fit_herd_array, axis=0)
                        confint_herd = 1.96*(std_values_herd[3]/np.sqrt(15))

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

        # Plot title
        if is_Pc_experiment:
            plt.title(f'{N_agents[0]} Traders, Pa = 0.0002, {N_time} Timesteps')
        elif is_Nagent_experiment:
            plt.title(f'Pa = 0.0002, Pc = 0.1, {N_time} Timesteps')
        elif is_Pa_Experiment:
            plt.title(f'{N_agents[0]} Traders, Pc = 0.1, {N_time} Timesteps')

        # Save image
        if image_dir != None:
            plt.savefig(image_dir+'priceseries.png')
        
        plt.show()


def vis_wealth_over_time(MarketObj, image_dir=None):

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
    for TraderObj in MarketObj.traders:
        ax1.plot(range(len(TraderObj.C)), TraderObj.C, alpha=0.2)
    ax2.hist([TraderObj.C[-1] for TraderObj in MarketObj.traders])

    if image_dir != None:
        plt.savefig(image_dir)
    plt.show()


def cluster_vis(MarketObj, t, cluster, image_dir=None):
    if cluster:

        ret = calc_norm_return(pd.DataFrame(MarketObj.p), False)
        # max_val, min_val =  max(MarketObj.avg_degree),  min(MarketObj.avg_degree)
        mean_val, std_val = np.mean(MarketObj.avg_degree),  np.std(MarketObj.avg_degree)
        # norm_degree = list(map(lambda x: (x - min_val)/(max_val - min_val), MarketObj.avg_degree))
        norm_degree = list(map(lambda x: (x - mean_val) / (std_val), MarketObj.avg_degree))

        plt.plot(np.linspace(0, t-1, t-1), ret.values, color="blue", label="Stock returns", linewidth=0.5)
        plt.plot(np.linspace(0, t, t), norm_degree, color="orange", label="Avg network degree")
        plt.xlabel("Time")
        # plt.ylabel("Normalized average network degree")
        plt.legend()

        if image_dir != None:
            plt.savefig(image_dir)
        plt.show()


def vis_vol_cluster(highp, window, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa_list, Pc_list, cluster, N):

    cluster_gaus, cluster_sp500 = vol_cluster(None, highp, window, N_time, True)
    count_gaus, bins_count_gaus = np.histogram(cluster_gaus, bins=[i for i in range(window+1)])
    count_sp500, bins_count_sp500 = np.histogram(cluster_sp500, bins=[i for i in range(window+1)])
    std_gaus = np.std(cluster_gaus)
    std_sp500 = np.std(cluster_sp500)

    standard_Pa = 0.0002
    standard_Pc = 0.1
    standard_Nagent = 100
    is_Pa_Experiment, is_Pc_experiment, is_Nagent_experiment = False, False, False

    # Perform visualisation for every pair
    for pair in itertools.combinations([standard_Pa, standard_Pc, standard_Nagent], 2):

        # Get the variable we want to vary
        is_Pa_Experiment = standard_Pa not in pair
        is_Pc_experiment = standard_Pc not in pair
        is_Nagent_experiment = standard_Nagent not in pair

        # Pa experiment
        if is_Pa_Experiment:
            Pc = [standard_Pc]
            agents = [standard_Nagent]
            Pa = Pa_list
            print("PA")
        # Pc experiment    
        elif is_Pc_experiment:
            Pa = [standard_Pa]
            agents = [standard_Nagent]
            Pc = Pc_list
            print("PC")
        # Agent experiment
        elif is_Nagent_experiment:
            Pc = [standard_Pc]
            Pa = [standard_Pa]
            agents = N_agents
            print("agent")

        # Image directory
        image_dir = f'images/Nagents{agents}_Pa{Pa}_Pc{Pc}/'
        # Make directory
        if not(os.path.isdir(image_dir)):
            os.mkdir(image_dir)

        # Loop over all configurations (in reality, 2 out of 3 are fixed and 1 real loop is performed)
        for a in Pa:
            for c in Pc:
                for agent in agents:
                    print(a)
                    print(c)
                    print(agent)
                    # Load all markets for this configuration
                    objects = management.loadMultipleMarkets(agent, N_time, C, A, p, garch, garch_n, garch_param, a, c, cluster, N)

                    cluster_measures_norm = []
                    cluster_measures_herd = []
                    bin_count_herd = []
                    count_herd = []
                    bin_count_norm = []
                    count_norm = []
                    for object in objects:
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
                        else:
                            cluster_measures_norm.append(cluster_measure)
                            bin_count_norm.append(bins_count_series[1:])
                            count_norm.append(count_series)

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
                        conf_herd = 1.96*(std_measure_herd/ np.sqrt(len(count_herd)))

                        # Plot graph line with label
                        if is_Pc_experiment:
                            plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, Pc = {c}, R = {round(mean_measure_herd,2)} $\\pm$ {round(conf_herd, 2)}")
                        elif is_Pa_Experiment:
                            plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, Pa = {a}, R = {round(mean_measure_herd,2)} $\\pm$ {round(conf_herd, 2)}")
                        elif is_Nagent_experiment:
                            plt.plot(bin_count_herd[0], mean_count_herd, label=f"Herd model, {agent} agents, R = {round(mean_measure_herd,2)} $\\pm$ {round(conf_herd, 2)}")

        plt.plot(bins_count_sp500[1:], count_sp500, label=f"SP500, R = {round(std_sp500/std_gaus, 2)}")
        plt.plot(bins_count_gaus[1:], count_gaus, label="Gaussian distribution")
        plt.xlabel("Number of trading days")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.legend()

        # Plot title
        if is_Pc_experiment:
            plt.title("Volatility clustering, 100 Agents, Pa = 0.0002")
        elif is_Nagent_experiment:
            plt.title("Volatility clustering, Pa = 0.0002, Pc = 0.1")
        elif is_Pa_Experiment:
            plt.title("Volatility clustering, 100 Agents, Pc = 0.1")

        # Save image
        if image_dir != None:
            plt.savefig(image_dir+'volclus.png')
        
        plt.show()


def plot_wealth_dist(MarketObj, image_dir=None):
    """
    Plots the wealth distribution alone
    """
    fig = plt.figure(figsize=(5, 5))

    sorted_wealth = sorted(
        MarketObj.traders, key=lambda x: x.A[-1]*MarketObj.p[-1] + x.C[-1])
    sorted_wealth = [i.A[-1]*MarketObj.p[-1] + i.C[-1] for i in sorted_wealth]
    cum_wealth = np.cumsum(sorted_wealth)

    # Determine distribution of wealth
    df = pd.DataFrame(sorted_wealth[30:])
    cdf, bins_count = create_cdf(df)

    # Fit power law
    model_array = np.array((cdf, bins_count[1:]))
    fit_comparison_array = np.zeros((len(bins_count), 5))

    best_fit_array = np.zeros(5)

    for i in range(1, len(bins_count)-1):

        x_values = np.log10(model_array[1, i:])
        y_values = np.log10(model_array[0, i:])

        fit_comparison_array[i-1, 0] = x_values[0]  # starting x_value for fit
        fit_comparison_array[i-1, 1] = x_values[-1]  # final x_value for fit
        fit_comparison_array[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1)  # fit line
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0, 1]
        rsquared = correlation_xy**2
        fit_comparison_array[i-1, 4] = correlation_xy**2  # add R^2 value to array

    best_fit_array[0] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 0] # starting x herd
    best_fit_array[1] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 1] # final x herd
    best_fit_array[2] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 2] # intercept herd
    best_fit_array[3] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 3] # slope herd

    plt.plot([10**best_fit_array[0], 10**best_fit_array[1]],
                [10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[0]),
                10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[1])],
             color='black', linestyle='--', label=f'Model fit, slope={round(best_fit_array[3],2)}')

    plt.scatter(bins_count[1:], cdf, s=6)
    plt.title('Wealth distribution after 10000 steps')
    plt.xlabel('Cumulative wealth')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.yscale('log')

    if image_dir != None:
        plt.savefig(image_dir+'wealth.png')
    else:
        plt.show()


def plot_lorenz_curve(objects, N_agents, image_dir=None):
    """
    Plots the Lorenz curve
    """
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

    all_t = [1000, 5000, 10000]

    all_sorted_wealth = {t: [] for t in all_t}
    all_cum_wealth = {t: [] for t in all_t}

    # N_time = len(MarketObj[0].p)
    # all_t = [int(0.1*N_time), int(0.5*N_time), N_time]

    for t in all_t:
        for MarketObj in [objects]:
            # Compute Lorenz curve and Gini coefficient
            sorted_wealth = sorted(MarketObj.traders, key=lambda x: x.A[t-1]*MarketObj.p[t-1] + x.C[t-1])
            sorted_wealth = [i.A[t-1]*MarketObj.p[t-1] + i.C[t-1] for i in sorted_wealth]
            cum_wealth = np.cumsum(sorted_wealth)

            all_sorted_wealth[t] += [sorted_wealth]
            all_cum_wealth[t] += [cum_wealth]

        sorted_wealth = np.average(all_sorted_wealth[t], axis=0)
        cum_wealth = np.average(all_cum_wealth[t], axis=0)

        # print(sorted_wealth)
        # print(cum_wealth)

        X = np.linspace(0, 1, len(MarketObj.traders))
        G = np.abs(1 - sum([(X[i+1]-X[i])*(cum_wealth[i+1]/sum(sorted_wealth)+cum_wealth[i]/sum(sorted_wealth))
                            for i in range(len(MarketObj.traders)-1)]))

        ax1.plot(np.linspace(0, 1,N_agents), cum_wealth/sum(sorted_wealth), label=f't={t}, Gini={round(G,2)}')

        # Determine distribution of wealth
        df = pd.DataFrame(sorted_wealth)
        cdf, bins_count = create_cdf(df)

        model_array = np.array((cdf, bins_count[1:]))
        fit_comparison_array = np.zeros((len(bins_count), 5))

        best_fit_array = np.zeros(5)

        # print(len(cdf), len(bins_count))

        j = 0

        for i in range(1, len(bins_count)-1):

            # x_values = np.log10(np.delete(model_array[1], np.where(model_array[1] < j)))
            # y_values = np.log10(np.delete(model_array[0], np.where(model_array[1] < j)))
            x_values = np.log10(model_array[1, i:])
            y_values = np.log10(model_array[0, i:])

            # print('y0', y_values[0])

            fit_comparison_array[i-1, 0] = x_values[0]  # starting x_value for fit
            fit_comparison_array[i-1, 1] = x_values[-1]  # final x_value for fit
            fit_comparison_array[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1)  # fit line
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation_xy = correlation_matrix[0, 1]
            rsquared = correlation_xy**2
            fit_comparison_array[i-1, 4] = correlation_xy**2  # add R^2 value to array
            # j = j + 0.01

        # print(fit_comparison_array)

        # print(f'Slope for best fit Gaus: {fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 3]}')
        best_fit_array[0] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 0] # starting x herd
        best_fit_array[1] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 1] # final x herd
        best_fit_array[2] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 2] # intercept herd
        best_fit_array[3] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 3] # slope herd

        ax2.plot([10**best_fit_array[0], 10**best_fit_array[1]],
                 [10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[0]),
                 10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[1])],
                 color='black', linestyle='--', label=f'Model fit, t={t}, slope={round(best_fit_array[3],2)}')
        ax2.scatter(bins_count[1:], cdf, label=f't={t}', s=6)

    ax1.plot([0, 1], [0,1], linestyle='--', color='black')
    ax1.set_title(f'Lorenz curve')
    ax1.set_xlabel('Cumulative share of agents')
    ax1.set_ylabel('Cumulative share of income')
    ax1.legend()

    ax2.set_title('Wealth distribution')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Magnitude of wealth')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()

    if image_dir != None:
        plt.savefig(image_dir+'lorenzTime.png')

    plt.show()


def plot_lorenz_curve_Nagents(objects, all_N_agents, image_dir=None):
    """
    Plots the Lorenz curve
    """

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))

    all_sorted_wealth = {n: [] for n in all_N_agents}
    all_cum_wealth = {n: [] for n in all_N_agents}

    for MarketObj in objects:

        N_agents = len(MarketObj.traders)

        # Compute Lorenz curve and Gini coefficient
        sorted_wealth = sorted(MarketObj.traders, key=lambda x: x.A[-1]*MarketObj.p[-1] + x.C[-1])
        sorted_wealth = [i.A[-1]*MarketObj.p[-1] + i.C[-1] for i in sorted_wealth]
        cum_wealth = np.cumsum(sorted_wealth)

        all_sorted_wealth[N_agents] += [sorted_wealth]
        all_cum_wealth[N_agents] += [cum_wealth]

    for N_agents in all_N_agents:

        sorted_wealth = np.average(all_sorted_wealth[N_agents], axis=0)
        cum_wealth = np.average(all_cum_wealth[N_agents], axis=0)

        # print(cum_wealth)
        X = np.linspace(0, 1, N_agents)
        G = np.abs(1 - sum([(X[i+1]-X[i])*(cum_wealth[i+1]/sum(sorted_wealth)+cum_wealth[i]/sum(sorted_wealth))
                            for i in range(N_agents-1)]))

        ax1.plot(np.linspace(0, 1,N_agents), cum_wealth/sum(sorted_wealth), label='$N_{agents}$ = '+str(N_agents)+', Gini='+str(round(G,2)))

        # Determine distribution of wealth
        df = pd.DataFrame(sorted_wealth)
        cdf, bins_count = create_cdf(df)

        model_array = np.array((cdf, bins_count[1:]))
        fit_comparison_array = np.zeros((len(bins_count), 5))

        best_fit_array = np.zeros(5)

        for i in range(1, len(bins_count)-1):

            # x_values = np.log10(np.delete(model_array[1], np.where(model_array[1] < j)))
            # y_values = np.log10(np.delete(model_array[0], np.where(model_array[1] < j)))
            x_values = np.log10(model_array[1, i:])
            y_values = np.log10(model_array[0, i:])

            # print('y0', y_values[0])

            fit_comparison_array[i-1, 0] = x_values[0]  # starting x_value for fit
            fit_comparison_array[i-1, 1] = x_values[-1]  # final x_value for fit
            fit_comparison_array[i-1, 2:4] = np.polynomial.polynomial.polyfit(x_values, y_values, deg=1)  # fit line
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation_xy = correlation_matrix[0, 1]
            rsquared = correlation_xy**2
            fit_comparison_array[i-1, 4] = correlation_xy**2  # add R^2 value to array
            # j = j + 0.01

        # print(fit_comparison_array)

        # print(f'Slope for best fit Gaus: {fit_comparison_array[np.argmax(fit_comparison_array[:,4]), 3]}')
        best_fit_array[0] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 0] # starting x herd
        best_fit_array[1] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 1] # final x herd
        best_fit_array[2] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 2] # intercept herd
        best_fit_array[3] = fit_comparison_array[np.nanargmax(fit_comparison_array[:, 4]), 3] # slope herd

        ax2.plot([10**best_fit_array[0], 10**best_fit_array[1]],
                 [10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[0]),
                 10**(best_fit_array[2]+best_fit_array[3]*best_fit_array[1])],
                 color='black', linestyle='--', label=f'Model fit, N={N_agents}, slope={round(best_fit_array[3],2)}')

        ax2.scatter(bins_count[1:], cdf, label='$N_{agents}$ = '+str(N_agents), s=6)

    ax1.plot([0, 1], [0,1], linestyle='--', color='black')
    ax1.set_title(f'Lorenz curve')
    ax1.set_xlabel('Cumulative share of agents')
    ax1.set_ylabel('Cumulative share of income')
    ax1.legend()

    ax2.set_title('Wealth distribution')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Magnitude of wealth')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()

    if image_dir != None:
        plt.savefig(image_dir+'lorenzAgents.png')
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

    # Even though this is single market, we loop over Nagents because we want to plot the lorenz curve and wealth distribution for various agents
    object_list = []
    for agents in N_agents:
            MarketObj = management.loadSingleMarket(agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i)
            object_list.append(MarketObj)

            # Plot Lorenz curve and wealth distribution, only for 100 agents
            if agents == 100:

                # Image directory
                image_dir = f'images/Nagents{agents}_Pa{Pa}_Pc{Pc}_i{i}/'

                # Make directory
                if not(os.path.isdir(image_dir)):
                    os.mkdir(image_dir)

                print("Visualise Lorenz curve over time")
                plot_lorenz_curve(MarketObj, agents, image_dir=image_dir)

    # Plot lorenz curve and wealth distribution for various agents
    print("Visualise Lorenz curve for various amounts of agents")
    image_dir = f'images/Nagents{N_agents[0]}_Pa{Pa}_Pc{Pc}_i{i}/'
    plot_lorenz_curve_Nagents(object_list, N_agents, image_dir=image_dir)

    # Other possible visualisations for a single market    
    MarketObj = management.loadSingleMarket(N_agents[0], N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, i)
    print("Visualise the wealth over time for a single run (100 agents)")
    vis_wealth_over_time(MarketObj, image_dir)
    print("Visualise clustering for singl run (100 agents)")
    cluster_vis(MarketObj, N_time, cluster, image_dir)


def visualiseMultipleMarketResults(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N):

    print("Multiple markets: visualise price series")
    vis_price_series(N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N)
    print("Multiple markets: visualise volatility clustering")
    vis_vol_cluster(0.2, 10, N_agents, N_time, C, A, p, garch, garch_n, garch_param, Pa, Pc, cluster, N)
