import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def avg_degree(MarketObj, cluster):
    """
    Determines the average degree of the random graph formed by the trader agents,
    saves average degree of current time step in market object
    @param MarketObj         Market object containing list of formed clusters
    @param cluster           Boolean to determine if model contains cluster formation
    """
    if cluster:
        degree_list = list(map(lambda x: len(x.members), MarketObj.clusters))
        MarketObj.avg_degree.append(np.mean(degree_list))

def calc_norm_return(df, abs):
    """
    Calculates standardized log returns of stock or index
    @param df        DataFrame containing closing prices of stock or index
    @param abs       Boolean to determine if absolute returns are required
    return:          DataFrame containing the standardized log returns
    """
    df = (np.log(df) - np.log(df.shift(1)))
    df = df.dropna()
    df = (df - df.mean())/df.std()
    if abs == True:
        df = df.abs()
    return df

def create_cdf(df):
    """
    Creates cumulative distribution of any data by utilizing histogram
    @param df       DataFrame containing desired data
    return:         Cumulative ditribution and histogram bin count
    """
    count, bins_count = np.histogram(df, bins=50)

    pdf = count / sum(count)
    pdf = np.flip(pdf)

    cdf = np.cumsum(pdf)
    cdf = np.flip(cdf)

    return cdf, bins_count

def fit_func(x, a, b):
    """
    Linear fit function used to fit onto a Power law distribution in form y = a + b*x
    @param a        float, constant
    @param b        float, constant
    @param x        float, input variable
    return:         a + b*x
    """
    return a + x * b

def curve_fit_log(xdata, ydata):
    """
    Fits linear curve to function in a loglog basis
    @param xdata        list of floats to serve as x data
    @param ydata        list of floats to serve as y data
    return:             list of fit params, list of parameter covariances, list of fitted y data
    """
    x_log = np.log10(xdata)
    y_log = np.log10(ydata)

    popt_log, pcov_log = curve_fit(fit_func, x_log, y_log)

    yfit_log = np.power(10, fit_func(x_log, *popt_log))

    return (popt_log, pcov_log, yfit_log)

def sample_gauss(N_time):
    """
    Creates Gaussian distributed prices, based on a random variable with mean 100 and variance 0.1,
    determines cumulative distribution of its standardized log returns.
    @param N_time       Integer length of time series
    return:             bins count of histogram, cumulative distribution
    """
    # creates Gaussian price series
    sample_list = list(map(lambda x: np.random.normal(100, 0.1), range(N_time)))
    series = pd.DataFrame(sample_list)
    # Determines standardized absolute returns
    series_norm = calc_norm_return(series, True)
    # create CDF
    cdf, bins_count = create_cdf(series_norm)

    return bins_count, cdf

def SP500_pl():
    """
    Creates cumulative distribution of SP500 returns over the past 30 years of closing data
    return:         histogram bin bount, cumulative distribution of SP500 returns
    """
    # retrieves data
    ticker = '^GSPC'
    df = yf.Ticker(ticker).history(period='30y', interval='1d')['Close']
    # standardized returns
    df = calc_norm_return(df, True)
    # cumulative distribution of returns and bin count of histogram
    cdf, bins_count = create_cdf(df)

    return bins_count, cdf

def vol_cluster(ret, highp, window, N_time, gauss):
    """
    Quantify volatility clustering based on method by Tseng and Li, method involves counting the number of days
    in a rolling window that exceed a certain return threshold.
    @param ret          list of float returns from the model
    @param highp        float between 0 and 1, percentage of highest returns to set as threshold
    @param window       integer, rolling window size
    @param N_time       integer, number of model iterations
    @param  gauss       boolean, if True method is applied to Guassian distributed returns and SP500 returns, else model returns
    return:             list of occurences where the return exceeded the threshold in the rolling window, equal to length N_time
    """
    if gauss:
        # gets Gaussian distribution price_returns_multiple_times
        sample_list = list(map(lambda x: np.random.normal(100, 0.1), range(N_time)))
        df = pd.DataFrame(sample_list)
        df = calc_norm_return(df, True)

        # defines threshold
        sample_sort = sorted(df.values, key=float)
        threshold_gaus = sample_sort[int((1-highp) * len(sample_sort))]

        # moves rolling window and counts occurences where threshold is exceeded
        cluster_gaus = [0 for _ in range(len(sample_list) - window)]
        for i in range(len(sample_list) - window):
            for j in range(window):
                if df.values[i+j] >= threshold_gaus:
                    cluster_gaus[i] += 1

        # repeats process for SP500 returns
        ticker = '^GSPC'
        df = yf.Ticker(ticker).history(period='30y', interval='1d')['Close']
        df = calc_norm_return(df, True)
        sp500_sort = sorted(df.values, key=float)
        threshold_sp500 = sp500_sort[int((1-highp) * len(sp500_sort))]

        cluster_sp500 = [0 for _ in range(len(df.values)- window)]
        for i in range(len(df.values) - window):
            for j in range(window):
                if df.values[i+j] >= threshold_sp500:
                    cluster_sp500[i] += 1

        return cluster_gaus, cluster_sp500

    else:
        # determines threshold for model returns
        ret_sorted = sorted(ret, key=float)
        threshold = ret_sorted[int((1-highp) * len(ret_sorted))]

        # moves rolling window and counts occurences where threshold is exceeded
        cluster_series = [0 for _ in range(len(ret)-window)]
        for i in range(len(ret)-window):
            for j in range(window):
                if ret[i+j] >= threshold:
                    cluster_series[i] += 1

        return cluster_series
