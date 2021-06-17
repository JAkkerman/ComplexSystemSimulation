import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calc_norm_return(df):
    df = (np.log(df) - np.log(df.shift(1)))
    df = df.dropna()
    df = (df - df.mean())/df.std()
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

def SP500_pl():
    ticker = '^GSPC'
    df = yf.Ticker(ticker).history(period='100y', interval='1d')['Close']
    

    df = calc_norm_return(df)

    cdf, bins_count = create_cdf(df)

    first_x, first_y = bins_count[1:18], cdf[:17]
    second_x, second_y = bins_count[17:], cdf[16:]
    total_x, total_y = bins_count[1:], cdf

    popt_first, pcov_first, real_first = curve_fit_log(first_x, first_y)
    popt_second, pcov_second, real_second = curve_fit_log(second_x, second_y)
    popt_total, pcov_total, real_total = curve_fit_log(total_x, total_y)

    plt.plot(first_x, real_first, label=f"$\\alpha$ = {round(popt_first[1],2)}")
    plt.plot(second_x, real_second, label=f"$\\alpha$ = {round(popt_second[1],2)}")
    plt.plot(total_x, real_total, label=f"$\\alpha$ = {round(popt_total[1],2)}")
    plt.scatter(bins_count[1:], cdf, color='black')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Normalized returns")
    plt.ylabel("Cumulative distribution")
    plt.show()
