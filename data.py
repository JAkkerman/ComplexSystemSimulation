import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def avg_degree(MarketObj, cluster):
    if cluster:
        degree_list = list(map(lambda x: len(x.members), MarketObj.clusters))
        MarketObj.avg_degree.append(np.mean(degree_list))

def calc_norm_return(df, abs):
    df = (np.log(df) - np.log(df.shift(1)))
    df = df.dropna()
    df = (df - df.mean())/df.std()
    if abs == True:
        df = df.abs()
    return df

def create_cdf(df):
    count, bins_count = np.histogram(df, bins=50)

    pdf = count / sum(count)
    pdf = np.flip(pdf)

    cdf = np.cumsum(pdf)
    cdf = np.flip(cdf)

    return cdf, bins_count

def fit_func(x, a, b):
    return a + x * b

def curve_fit_log(xdata, ydata) :
    x_log = np.log10(xdata)
    y_log = np.log10(ydata)

    popt_log, pcov_log = curve_fit(fit_func, x_log, y_log)

    yfit_log = np.power(10, fit_func(x_log, *popt_log))

    return (popt_log, pcov_log, yfit_log)

def sample_gauss(N_time):
    sample_list = list(map(lambda x: np.random.normal(100, 0.1), range(N_time)))
    series = pd.DataFrame(sample_list)
    series_norm = calc_norm_return(series, True)
    cdf, bins_count = create_cdf(series_norm)
    popt, pcov, real = curve_fit_log(bins_count[1:], cdf)
    # print(cdf[:-20])
    # coeffs = np.polyfit(bins_count[1:-5], cdf[:-5], deg=4)
    # real = list(map(lambda x: coeffs[0]*x**8 + coeffs[1]*x**7 + coeffs[2]*x**6 + coeffs[3]*x**5 + coeffs[4]*x**4 + coeffs[5]*x**3 + coeffs[6]*x**2 + coeffs[7]*x + coeffs[8], bins_count[1:-5]))
    # real = list(map(lambda x: coeffs[0]*x**4 + coeffs[1]*x**3 + coeffs[2]*x**2 + coeffs[3]*x +coeffs[4], bins_count[1:-5]))


    return bins_count, cdf

def SP500_pl():
    ticker = '^GSPC'
    df = yf.Ticker(ticker).history(period='30y', interval='1d')['Close']


    df = calc_norm_return(df, True)

    cdf, bins_count = create_cdf(df)

    first_x, first_y = bins_count[1:18], cdf[:17]
    second_x, second_y = bins_count[17:], cdf[16:]
    total_x, total_y = bins_count[1:], cdf

    popt_first, pcov_first, real_first = curve_fit_log(first_x, first_y)
    popt_second, pcov_second, real_second = curve_fit_log(second_x, second_y)
    popt_total, pcov_total, real_total = curve_fit_log(total_x, total_y)

    # plt.plot(first_x, real_first, label=f"$\\alpha$ = {round(popt_first[1],2)}")
    # plt.plot(second_x, real_second, label=f"$\\alpha$ = {round(popt_second[1],2)}")
    # plt.plot(total_x, real_total, label=f"$\\alpha$ = {round(popt_total[1],2)}")
    # plt.scatter(bins_count[1:], cdf, color='black')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    # plt.xlabel("Normalized returns")
    # plt.ylabel("Cumulative distribution")
    # plt.show()

    return bins_count, cdf

def vol_cluster(ret, highp, window, N_time, gauss):
    if gauss:
        sample_list = list(map(lambda x: np.random.normal(100, 0.1), range(N_time)))
        df = pd.DataFrame(sample_list)
        df = calc_norm_return(df, True)
        sample_sort = sorted(df.values, key=float)
        threshold_gaus = sample_sort[int((1-highp) * len(sample_sort))]

        cluster_gaus = [0 for _ in range(len(sample_list) - window)]
        for i in range(len(sample_list) - window):
            for j in range(window):
                if df.values[i+j] >= threshold_gaus:
                    cluster_gaus[i] += 1

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
        ret_sorted = sorted(ret, key=float)
        threshold = ret_sorted[int((1-highp) * len(ret_sorted))]

        cluster_series = [0 for _ in range(len(ret)-window)]
        for i in range(len(ret)-window):
            for j in range(window):
                if ret[i+j] >= threshold:
                    cluster_series[i] += 1

        return cluster_series



    # std_series = np.std(cluster_series)
    # std_gaus = np.std(cluster_gaus)
    #
    # cluster_measure = std_series/std_gaus
    #
    # return cluster_series, cluster_gaus, cluster_measure
