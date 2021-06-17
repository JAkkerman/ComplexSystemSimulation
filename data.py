import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

ticker = '^GSPC'
df = yf.Ticker(ticker).history(period='100y', interval='1d')['Close']
df = (np.log(df) - np.log(df.shift(1)))
df = df.dropna()
df = (df - df.mean())/df.std()
df = df.abs()
# df = (df-df.min())/(df.max()-df.min())

# print(df)

# sort_ret = sorted(df.values)
# sort_95 = sort_ret[int(len(sort_ret)*0.95):]

# print(len(sort_95))

# count, bins_count = np.histogram(sort_95, bins=100)
count, bins_count = np.histogram(df, bins=30)
print(count[10:], bins_count[10:])

pdf = count / sum(count)
pdf = np.flip(pdf)

cdf = np.cumsum(pdf)
cdf = np.flip(cdf)

# plt.hist(df, bins=100)
plt.loglog(bins_count[1:], cdf)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(20)
plt.show()
