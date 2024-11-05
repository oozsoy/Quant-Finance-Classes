import yfinance as yf
import numpy as np
import pandas as pd 
from scipy.stats import skew, kurtosis
from pypfopt.efficient_frontier import EfficientFrontier

class asset:
    
    '''Class to download and process asset price data
       TO DO: More methods for preprocessing, statistics, plotting and mean-variance optimization etc.. '''
    def __init__(self,stocks,start,end):
        
        self.stocks = stocks 
        self.start = start 
        self.end = end 
        
        self.price_data = None
        self.return_df = None
        self.log_return_df = None
        self.weigths = None
        
    def download(self):
        
        self.price_data = yf.download(self.stocks, self.start, self.end)
        
        if 'Close' not in self.price_data.columns:
            
            raise ValueError('Price data download failed or the ticker does not have close price data for the specified dates')
        
        else: 
            
            print(f'Price data downloaded for {self.stocks} from {self.start} to {self.end}')
            
    def to_returns(self):
        
        '''Method to compute simple and log returns'''
        
        if self.price_data is None:
            
            raise ValueError('"Price data not downloaded. First download price data using .download() method')
            
             
        self.log_return_df =  np.log(1 + self.price_data['Close'].pct_change()).dropna() 
        self.return_df = self.price_data['Close'].pct_change().dropna()
            
        return self.return_df, self.log_return_df
        
    def summary_stats(self):
        
        '''Get descriptive statistics of returns, adding skewness and kurtosis to the list'''
        
        if self.return_df is None:
            
            raise ValueError('Returns are not calculated, first run .to_returns()')
            
        # calculate stats on log returns
        stats_df = self.log_return_df.describe() 
            
        stats_df.columns = self.log_return_df.columns + '_r'
         
        stats_df.loc['skew'] = skew(self.log_return_df)
        # fisher flag false --> # kurtosis is not w.r.t normal dist. where kurtosis = 3.
        stats_df.loc['kurtosis'] = kurtosis(self.log_return_df, fisher = False) 
                                                                        
        return stats_df
    
    def get_allocations(self, type = 'max_SR'):
        
        '''Get the asset allocations for a given return dataframe and a specified risk/reward preference'''
        
        # historical mean returns and covariance for the specified time frame
        mean_returns = self.return_df.mean()
        covariance = self.return_df.cov()
        
        # Initialize the efficient frontier class from PyPortfolioOpt
        ef = EfficientFrontier(mean_returns, covariance, weight_bounds=(0.01,0.5))
        
        if type == "max_SR":
            
            # get max sharpe ratio allocation weights: a dictionary with ticker names as keys
            ef.max_sharpe(risk_free_rate=0.)
            max_sr_weights = ef.clean_weights()  

            # Render the allocations as an array
            self.weights = np.array([val for key, val in max_sr_weights.items()])
            
            return self.weights
    
    def get_portfolio_returns(self):
        
        ''' Output portfolio returns using the simple return dataframe'''
        if self.weights is None:
            raise TypeError('Calculate allocations for the assets first, using .get_allocations()')
                
        return (self.return_df * self.weights).sum(axis = 1)
        
        
    def mc_sim(self, mc_sims, current_pval = 1., plot = True):
        
        if self.weights is None:
            
            raise ValueError('Compute the allocations first using get_allocations()')
        
        self.mc_sims = mc_sims # number of simulations
        T = 100 # time horizon in days 

        mean_returns = self.return_df.mean()
        covariance = self.return_df.cov()
        
        #Cholesky decomposition of covariance matrix
        L = np.linalg.cholesky(covariance)
        
        # mean returns of assets, T x D, D = number of assets
        mean_Mat = np.full(shape = (T,len(self.weights)), fill_value=mean_returns)
        
        # variable to store portfolio simulations, T x number of simulations
        portfolio_sims = np.full(shape = (T,self.mc_sims), fill_value=0.0)

        # current portfolio value, default is set to 1.
        initial_portfolio_val = current_pval 

        for sim_id in range(self.mc_sims):
            
            #sample random (uncorrelated) variables 
            Z = np.random.normal(size = (T, len(self.weights)))
            
            # generate correlated log returns of assets
            daily_log_ret = mean_Mat + np.inner(Z,L) # L.Z is T x D
            
            # returns
            daily_ret = np.exp(daily_log_ret) - 1

            # for each simulation compute the price evolution for each T
            portfolio_sims[:,sim_id] = np.cumprod(1 + np.inner(daily_ret,self.weights)) * initial_portfolio_val
            
        if plot == True:
            
            fig, axes = plt.subplots(figsize = (9,5))

            axes.plot(portfolio_sims, alpha = 0.04, c = 'black')

            axes.set_ylabel('Portfolio Value')
            axes.set_xlabel('time [days]')

            axes.set_title('MC simulation of portfolio value')
            axes.grid()
            
        # return only the price distribution at the final time      
        return portfolio_sims[-1,:]      
