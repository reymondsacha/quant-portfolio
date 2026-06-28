import numpy as np
from scipy.stats import norm

class BlackScholesAnalytical:
    def __init__(self, S, K, T, r, sigma):
        self.S = np.asarray(S, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)

        epsilon = 1e-8
        self.T = np.maximum(np.asarray(T, dtype=float), epsilon)

    @property
    def d1(self):
        return (np.log(self.S / self.K)  + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """
        Calculates the analytical Black-Scholes call price.
        """
        C = self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)  
        return C

    def call_delta(self):
        """ 
        Calculates the analytical Black-Scholes call delta.
        """
        return norm.cdf(self.d1)

 

    def call_gamma(self):
        """
        Calculates the analytical Black-Scholes gamma.
        """
        gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma

    def call_vega(self):
        """
        Calculates the analytical Black-Scholes vega.
        """
        vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        return vega


    def call_theta(self, daily=False):
        """
        Calculates the analytical Black-Scholes theta.

        daily(bool): If True, returns daily theta (annual/365)
        """
        term_1 = - (self.S * norm.pdf(self.d1) * self.sigma) / (2* np.sqrt(self.T))
        term_2 = - self.K * self.r * np.exp(-self.r * self.T) * norm.cdf(self.d2)

        theta = term_1 + term_2

        if daily:
            return theta/365

        return theta

    def call_vanna(self):
        """
        Calculates the analytical Black-Scholes vanna.
        """
        vanna = - norm.pdf(self.d1) * (self.d2/self.sigma)
        return vanna

    def call_charm(self):
        """
        Calculates the analytical Black-Scholes charm.
        """
        charm = - norm.pdf(self.d1) * ((self.r / (self.sigma * np.sqrt(self.T))) - (self.d2 / (2 * self.T)))
        return charm

        

