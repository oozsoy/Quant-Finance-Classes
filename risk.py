class risk:
    
    ''' Class to implement VaR and CVaR computation'''
    
    def __init__(self, returns, dt, in_portfolio_val, fin_portfolio_val):
        
        ''' dt: time horizon for historical var/cvar method'''
        # initialize with a return series or dataframe
        if not isinstance(returns, (pd.Series, pd.DataFrame)):
            
            raise TypeError('Input returns must be pd.Series or pd.DataFrame object')  
        
        self.dt = dt
        # Use log-returns as input for portfolio or assets
        self.returns = returns
        # initial investment / portfolio value
        self.in_portfolio_val = in_portfolio_val
        # portfolio value at the final time slice of various simulations
        self.portfolio_val = fin_portfolio_val
        
    def historical_var(self, alpha = 5):
        
        if isinstance(self.returns, pd.Series):
            
            var_lr = -np.percentile(self.returns, alpha)
            
            return self.in_portfolio_val * var_lr * np.sqrt(self.dt)
        
        elif isinstance(self.returns, pd.DataFrame):
            
            var_lr = -self.returns.aggregate(lambda x: np.percentile(x, alpha))
            
            return self.in_portfolio_val * var_lr * np.sqrt(self.dt)
        
    def historical_cvar(self, alpha = 5):
        
        neg_var = - self.historical_var(alpha)/(np.sqrt(self.dt) * self.in_portfolio_val)
                      
        if isinstance(self.returns, pd.Series):
                                    
            below_var_returns = - self.returns[self.returns <= neg_var]
            
            return below_var_returns.mean() * np.sqrt(self.dt) * self.in_portfolio_val
                   
        elif isinstance(self.returns, pd.DataFrame):
            
            var = -self.returns.apply(lambda x: x[x <= neg_var[x.name]].mean())
            
            return var * np.sqrt(self.dt) * self.in_portfolio_val
        
    def mc_var(self, alpha = 5):
        
        ''' Read portfolio value at the final time of the simulations 
            to return its percentile at a given confidence level alpha'''
               
        # validate the input
        if isinstance(self.portfolio_val, np.ndarray):
            
            # maximum loss at 1-alpha CL: initial_investment - worst_final_portfolio_value @ 1-alpha CL
            return - np.percentile(self.portfolio_val, alpha) + self.in_portfolio_val 
    
        else:
            
            raise TypeError('Input must be a np.array')
    

    def mc_cvar(self, alpha = 5):
        
        ''' Read portfolio value to output the expected shortfall (CVaR) for a given confidence level alpha '''
        var_pval_fin = -(self.mc_var(alpha)-self.in_portfolio_val)
        
        # validate the input
        if isinstance(self.portfolio_val, np.ndarray):
            
            below_var_bool = self.portfolio_val <= var_pval_fin
            
            return -self.portfolio_val[below_var_bool].mean() + self.in_portfolio_val
    
        else:
            
            raise TypeError('Input must be a np.array')
