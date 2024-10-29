import yfinance as yf
import numpy as np
import pandas as pd 
from scipy.stats import skew, kurtosis

class finance_data:
    
    '''Class to download and process asset price data
       TO DO: More methods... '''
    def __init__(self,stocks,start,end):
        
        self.stocks = stocks 
        self.start = start 
        self.end = end 
        
        self.price_data = None
        self.return_df = None
        
    def download(self):
        
        self.price_data = yf.download(self.stocks, self.start, self.end)
        
        if 'Close' not in self.price_data.columns:
            
            raise ValueError('Price data download failed or the ticker does not have close price data for the specified dates')
        
        else: 
            
            print(f'Price data downloaded for {self.stocks} from {self.start} to {self.end}')
            
    def to_returns(self, log = True):
        
        '''Method to compute either simple returns or log returns'''
        
        if self.price_data is None:
            
            raise ValueError('"Price data not downloaded. First download price data using .download() method')
            
        if log: 
            
            self.return_df =  np.log(1 + self.price_data['Close'].pct_change()).dropna() 
        
        else:
            
            self.return_df = self.price_data['Close'].pct_change().dropna()
            
        return self.return_df
        
    def summary_stats(self):
        
        '''Get descriptive statistics of returns, adding skewness and kurtosis to the list'''
        
        if self.return_df is None:
            
            raise ValueError('Returns are not calculated, first run .to_returns()')
            
        stats_df = self.return_df.describe() 
            
        stats_df.columns = self.price_data['Close'].columns + '_R'
         
        stats_df.loc['skew'] = skew(rets)
        # fisher flag false --> # kurtosis is not w.r.t normal dist. where kurtosis = 3.
        stats_df.loc['kurtosis'] = kurtosis(rets, fisher = False) 
                                                                        
        return stats_df
