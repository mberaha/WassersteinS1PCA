import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from .distribution import PeriodicDistribution

class PeriodicTransportMap(object):
    def __init__(self, init_point: PeriodicDistribution):
        self.init_point = init_point

    def compute(self, end_point: PeriodicDistribution):
        self.end_point = end_point
        self.theta = self.get_opt_theta()
        self._map = lambda x: self.end_point.periodic_quantile(
            self.theta, self.init_point.periodic_cdf(0, x))

    def __call__(self, transport_grid):
        return self._map(transport_grid)

    def get_pushforward_measure(self):
        grid = np.linspace(-1, 2, 1000)
        trans_eval = self.__call__(grid)
        return self.get_pushforward_from_transport(grid, trans_eval)

    def get_pushforward_from_transport(self, grid, trans_eval):
        trans_inv = interp1d(trans_eval, grid, kind="linear")
        
        new_grid = np.linspace(np.min(trans_eval), np.max(trans_eval), 1000)
        cdf_eval = self.init_point.periodic_cdf(0, trans_inv(new_grid))
        
        zero_idx = np.sum(new_grid < 0)
        cdf_eval = cdf_eval - cdf_eval[zero_idx]

        wh = np.where((new_grid >= 0) & (new_grid <= 1))
        new_grid = new_grid[wh]
        cdf_eval = cdf_eval[wh]
        cdf_eval = cdf_eval - cdf_eval[0]
        out = PeriodicDistribution(supp_grid=new_grid, cdf_eval=cdf_eval)
        return out

    def get_opt_theta(self):
        quant_grid = np.linspace(0, 1, 1000)
        theta_grid = np.linspace(-1, 1, 250)
        shifted_quantes = self.end_point.periodic_quantile(theta_grid, quant_grid)
        integrands = (shifted_quantes - 
                      self.init_point.periodic_quantile(0, quant_grid))**2
        wdists = trapz(integrands, quant_grid)
        return theta_grid[np.argmin(wdists)]

    def wdist(self, theta, quant_grid):
        return trapz(
            (self.init_point.periodic_quantile(0, quant_grid) - 
            self.end_point.periodic_quantile(theta, quant_grid))**2, quant_grid)

    def compute_spline_expansion(self, spbasis):
        self.spbasis = spbasis
        t_eval = self.__call__(spbasis.xgrid)
        self.spline_coeffs = self.spbasis.get_spline_expansion(t_eval)
