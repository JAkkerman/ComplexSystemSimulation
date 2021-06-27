# Complex System Simulation - Financial Phenomena in Genoa market model



Group 14

Authors: Joos Akkerman, Oscar de Keijzer, Susy Maijer & Loek van Steijn



## Introduction
In this project we adopt the Genoa Market model by Raberto et al. [[1]](#1) to determine the influence of cluster/group formation amongst individual traders (i.e. Hedge Funds or groups of private investors) on commonly found emergent phenomena in financial markets. The model consists of a set of agents, which perform financial transactions among each other. As agents determine their limit price and desired quantity to be sold or bought, an equilibrium price is found that clears the market. During this process traders are linked together to form a cluster with probability $P_a$, these clusters can merge with other clusters if two agents from two different clusters are linked together. Every model iteration a cluster is randomly activated with probability $P_c$, when a cluster is activated each trader therein will either cumulatively want to sell or buy the asset after which the cluster is deleted. Over time, this generates a price series of which we analyse the commonly found emergent phenomena and compare it to the real-world financial time series of the S&P500 index and a Gaussian distributed return time series.

We primarily focus on the "cubic law of stock returns" [[2]](#2), which dictates that the fat tail of stock returns can be approximated as a power law distribution with an exponent of $\sim 3$. Secondly, we focus on the volatility clustering [[3]](#3), which states that large returns tend be followed by large returns and small returns tend to be followed by small returns. Lastly, we focus on the wealth distribution amongst individual traders, this is supposed to be Pareto distributed and can be summarized that roughly 80% of the total wealth is held by 20% of the total population.


## Navigation
Overview of all model elements and how to navigate them.

## Main
In this file the main model is executed and the main model parameters can be adjusted here too. The parameters include:

* `N_time`: integer, number of model iterations
* `cluster`: boolean, True if traders are allowed to form clusters, else Traders act independently
* `C`: integer, amount of cash each trader agent is initialized with
* `A`: integer, number of stocks each trader agent is initialized with
* `p`: integer, starting price of stock
* `garch`: boolean, True if volatility of stock is forecasted with Garch model
* `garch_n`: integer, Used in data storage
* `garch_param`: list of two intgers, Garch(p,q) degrees, standard [1,1]
* `Pa_list`: list of floats, contains different probabilities of two traders forming a cluster
* `Pc_list`: list of floats, contains different probabilities of a cluster activating where all traders therein cumulatively buy or sell stock
* `N_agents`: list of integers, contains different number of agents in model
* `N_concurrent`: integer, how many times each simulation is performed, increases statistical relevance
* `N_cores`: integer, Number of CPU cores maximally used for data collection


### Classes
Overview of all model classes.

#### Trader
The Trader class contains the data and methods for the individual trader objects. The following data is stored:

* `id`: id of the trader
* `market`: object, refers to the market object the trader is in.
* `C`: list of floats, time series of the cash owned by the trader
* `A`: list of floats, time series of assets owned by the trader
* `a_b`: float, quantity of assets the trader wants to buy at time `t`
* `b_i`: float, limit price trader wants to buy at (so highest possible price, will buy for a lower price in most cases).
* `a_s`: float, quantity of assets the trader wants to sell at time `t`
* `s_i`: float, limit price trader wants to sell at (so lowest possible price, will sell for a higher price in most cases).
* `P_i`: float, buy order probability. This determines if the trader will set a buy or sell order.
* `in_cluster`: object, refers to cluster object, if the trader is in a cluster, otherwise `None`.

The Trader object contains the following methods:

* `trade_decision()`: decides if trader will place buy or sell order, based on probability `P_i`. Then refers to `buy()` or `sell()`.
* `buy()`: determines `a_b` and `b_i`, places buy order.
* `sell()`: determines `a_s` and `s_i`, places sell order.
* `no_trade()`: if the market price does not meet the limit price, no trade is made. This function then extends the series for `A` and `C` with their respective values at time `t`.

#### Market
The Market object contains the methods that allow the market mechanism to function. It also tracks the macro variables of the market. The following data is stored:

* `p`: list of floats, time series of the market price.
* `cluster`: boolean, determines if clustering is activated.
* `traders`: list of objects, containing all trader objects
* `buyers`: list of objects, containing all buyers at time `t`, emptied after time step is over.
* `sellers`: list of objects, containing all sellers at time `t`, emptied after time step is over.
* `clusters`: list of objects, containing all clusters at time `t`. Updated if clusters are added or deleted.
* `Pc`: float, probability of clusters activating
* `Pa`: float, probability of clusters forming
* `hist_vol`: float, historical volatility. updated based on generated time series.
* `sigma`: list of floats, time series of true volatility, as the historical volatility is multiplied by a constant $3.5$, in accordance with [[1]](#1). Alternatively, this can be done using an implementation of a GARCH model (which was not used in the final results).
* `avg_degree`: list of floats, average degree of clusters.
* `garch`: boolean, governs whether a GARCH model is used to determine `sigma`.
* `garch_param`: list of integers, GARCH(p,q) degrees ([p,q]).
* `garch_model`: function, fitted GARCH model for time step `t`.

The market objects contains the following methods that determine the **pricing process**:

* `get_equilibrium_p()`: uses limit prices and quantities set by traders to determine the equilibrium price (for which supplied quantity equals demanded quantity). Utilizes `find_intersection()` to find intersection between supply and demand curves.
* `find_intersection()`: Finds set of buyers and sellers for which the market is cleared to the largest extend, and for which the determined equilibrium price satisfies the set limit price.
* `perform_transactions()`: receives true buyers and sellers, performs transactions by updating `A` and `C` for true buyers and sellers.

The market objects contains the following methods that determine the **volatility estimates**:

* `update_sigma()`: updates `sigma` series, using the updated `hist_vol` and either the method by [[1]](#1) or the fitted GARCH model.
* `update_hist_vol()`: updates `hist_vol` using the log-returns of the generated price series.
* `fit_GARCH()`: fits GARCH model to previous volatility and return series, ranging 20-100 time steps back, depending on availability of data. Determines volatility for time step `t`.

The market objects contains the following methods that determine the **clustering Process**:
* `form_clusters()`: decides which clusters are to be formed. Randomly chooses two traders each time step to be clustered. After which, it refers to the following functions based on the cluster state of the traders:
    * `init_cluster()`: if both traders are not in a cluster, form new cluster object. Update `clusters`.
    * `merge_cluster()`: if both traders are in a cluster, merge both clusters to a new cluster object. Update `clusters`.
    * If one of the traders is in a cluster and the other is not, add second trader to cluster (through method in `Cluster` object of first trader).

* `activate_cluster()`: based on `Pc`, activates one of the randomly chosen clusters.

Finally, the Market object contains the method `reset_lists()`, which clears the `buyers` and `sellers` lists.

#### Cluster
The Cluster class contains the data and methods for the cluster objects. The following data is stored:

* `members`: List of objects, refers to Trader objects that are part of the cluster.
* `market`: Object, refers to the Market object the cluster is part of.

The cluster object contains the following methods:
* `add_to_cluster()`: Adds a Trader object to the cluster.
* `activate()`: Activates cluster by setting all probabilities `P_i` to either 1 or 0 for all traders in the cluster.
* `self_destruct()`: After being activated, removes Cluster object from traders and Market object.


### Data
The file `data.py` contains some general utility functions

* `avg_degree()`: Calculates the average degree of the randomly formed clusters of traders at each model iteration
* `calc_norm_return()`: Calculates either the regular or absolute standardized log returns of financial time series
* `create_cdf()`: Creates cumulative distribution of any data by utilizing histograms
* `fit_func()`: Linear fit function of shape $y = a\cdot x + b$
* `curve_fit_log()`: Fits linear curve to function in a loglog basis
* `sample_gauss()`: Creates Gaussian distributed prices, based on a random variable with mean 100 and variance 0.1
* `SP500_pl()`: Creates cumulative distribution of SP500 returns over the past 30 years of closing data
* `vol_cluster()`: Quantify volatility clustering based on method by Tseng and Li [[4]](#4), method involves counting the number of days
    in a rolling window that exceed a certain return threshold.

### Management
The file `management.py` contains files used to store and retreive data

* `getResultsDirectoryPath()`: Returns the directory of the results of a specific run configuration.
* `makeDirectories()`: Make directories for each configuration parameter (done in advance due to concurrency problems which otherwise arise)
* `saveSingleMarket()`: Write the final Market Object to disk.
* `loadSingleMarket()`: Load a Market Object from disk.
* `loadMultipleMarkets()`: Load all N Market Objects from a single parameter configuration.


### Visualization

The file `visualization.py` contains the functions used to plot the results. These are the following:

* `vis_market_cross()`: visualizes the market equilibrium cross.![](https://i.imgur.com/tVkXxlr.png)

* `vis_price_series()`: Visualizes the return distribution of the price series, fits a power law to the (tail of the) distribution, shows the fitted curve with the best fit.
* `vis_wealth_over_time()` Visualizes the individual wealth time series of all agents, and the histogram of wealth at the final time step.
* `cluster_vis()`: visualizes the average degree of the formed clusters over time.![](https://i.imgur.com/OEmkR3F.png)

* `vis_vol_cluster()`: Visualizes volatility clustering graphs based on the method by Tseng and Li [[4]](#4) for multiple Pa, Pc and number of agents.![](https://i.imgur.com/gbE5JSB.png)


* `plot_wealth_dist()`: Visualizes the wealth distribution of single iteration.
* `plot_lorenz_curve()`: Visualizes the Lorenz curve and wealth distribution, averaged over multiple iterations for increasing time steps. This is to show the development of wealth distribution over time.![](https://i.imgur.com/3FyjvQy.png)


* ` plot_lorenz_curve_Nagents()`: Visualizes the Lorenz curve and wealth distribution, averaged over multiple iterations for increasing amount of agents. This is to show the development of wealth distribution as the number of agents increases.![](https://i.imgur.com/VcqQerQ.png)
* `vis_volatility_series()`: Visualizes time series of asset volatility.
* `visualiseSingleMarketResults()`: Calls plot functions to show the results of a single run.
* `visualiseMultipleMarketResults()`: Calls plot functions to show the results of multiple averaged runs.


## Operation
Before running any code it is important to update your python modules by installing requirements.txt found in the main folder. Ensure that you have Python and Pip installed. You can download Python [here](https://www.python.org/downloads/). You can install the neccessary Python modules by running:

```command
pip install -r requirements.txt
```

Once the correct packages are installed, any parameters can be changed in the `main.py` file as described above. The model can be executed to either collect data by running

```command
python main.py collect
```
This will save all objects to pickle. These files can be visualized by running

```command
python main.py visualize
```
This will visualise the collected data as described above.

## References

<a id="1">[1]</a> Raberto, M., Cincotti, S., Focardi, S. M., & Marchesi, M. (2001). Agent-based simulation of a financial market. _Physica A: Statistical Mechanics and its Applications_, _299_(1-2), 319-327.
<a id="2">[2]</a> Gopikrishnan, P., Plerou, V., Amaral, L.A.N., Meyer, M., Stanley, H.E., (1999) Scaling of the distribution of fluctuations of financial market indices. The American Physical Society Volume 60, Number 5
<a id="3">[3]</a> Mandelbrot, B., 1963. The variation of certain speculative prices. Journal of Business 36 (4), 394–419.
<a id="4">[4]</a> Tseng, J. J., & Li, S. P. (2012). Quantifying volatility clustering in financial time series. International Review of Financial Analysis, 23, 11-19.
