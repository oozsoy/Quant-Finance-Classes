class risk:
    
    ''' Class to implement traditional VaR and CVaR computations
        TO DO: Include Monte Carlo method and Delta-Normal (variance-covariance) method'''
    
    def __init__(self, returns):
        # initialize with a return series or dataframe
        if not isinstance(returns, (pd.Series, pd.DataFrame)):
            
            raise TypeError('Input returns must be pd.Series or pd.DataFrame object')  
        
        self.returns = returns
        
    def historical_var(self, alpha = 5):
        
        if isinstance(self.returns, pd.Series):
            
            return -np.percentile(self.returns, alpha)
        
        elif isinstance(self.returns, pd.DataFrame):
            
            return -self.returns.aggregate(lambda x: np.percentile(x, alpha))
        
    def historical_cvar(self, alpha = 5):
        
        neg_var = - self.historical_var(alpha)    
                      
        if isinstance(self.returns, pd.Series):
                                    
            below_var_returns = self.returns[self.returns <= neg_var]
            
            return -below_var_returns.mean()
                   
        elif isinstance(self.returns, pd.DataFrame):
            
            return -self.returns.apply(lambda x: x[x <= neg_var[x.name]].mean())
