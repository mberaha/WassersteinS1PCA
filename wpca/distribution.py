from argparse import ArgumentError
import logging
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import cumtrapz, simps
from scipy.ndimage import gaussian_filter1d

from .spline import MonotoneQuadraticSplineBasis


class PeriodicDistribution(object):
    def __init__(self, supp_grid=None, cdf_eval=None, 
                 quant_grid=None, quant_eval=None):
        if cdf_eval is None and quant_eval is None:
            raise ArgumentError("At least one between cdf_eval and quant_eval"
                                " must be provided")

        if supp_grid is not None:
            self.init_from_cdf(supp_grid, cdf_eval)
        else:
            self.init_from_quantile(quant_grid, quant_eval)

        self.pdf = None

    def cdf(self, x):
        out = self._cdf(x)
        out[x < self.sa] = 0
        out[x > self.sb] = 1
        return out

    def quantile(self, t):
        out = self._quantile(t)
        out[t < self.qa] = self.sa
        out[t > self.qb] = self.sb
        return out

    def periodic_quantile(self, theta, t):
        if type(theta) is np.ndarray:
            t_min_theta = t[np.newaxis, :] - theta[:, np.newaxis]
        else:
            t_min_theta = t - theta
        out = self.quantile(t_min_theta - np.floor(t_min_theta))
        out += np.floor(t_min_theta)
        return out

    def periodic_cdf(self, theta, x):
        out = np.zeros_like(x)
        wh = np.where((x > 0) & (x <= 1))
        out[wh] = self.cdf(x[wh]) + theta
        for p in np.arange(int(np.min(x)) - 1, int(np.max(x)) + 1):
            wh = np.where((x > p) & (x <= p+1))
            out[wh] = self.cdf(x[wh] - p) + theta + p
        return out

    def init_from_cdf(self, supp_grid, cdf_eval):
        self.supp_grid = supp_grid
        self.cdf_eval = cdf_eval 

        keep = np.where(np.diff(self.cdf_eval) > 1e-20)
        self.sa = np.min(supp_grid[keep])
        self.sb = np.max(supp_grid[keep])
        self.qa = np.min(cdf_eval[keep])
        self.qb = np.max(cdf_eval[keep])

        self._cdf = interp1d(supp_grid, cdf_eval, kind="linear", 
                             fill_value="extrapolate")
        
        self._quantile = interp1d(cdf_eval[keep], supp_grid[keep], kind="linear",
                                  fill_value="extrapolate")
        self.quant_grid = np.linspace(0, 1, 1000)
        self.quant_eval = self.quantile(self.quant_grid)

    def compute_quantile_spline_expansion(self, wbasis):
        self.wbasis = wbasis
        self.quantile_coeffs = self.wbasis.get_spline_expansion(
            self.quant_eval, self.quant_grid)

    def eval_smooth_pdf(self, grid, smooth_param=0.005):
        if self.pdf is None:
            ext_grid = np.linspace(-1, 2, 3000)
            cdf = self.periodic_cdf(0, ext_grid)
            cdf_spline = UnivariateSpline(ext_grid, cdf, s=smooth_param)
            self.pdf = cdf_spline.derivative()
        return self.pdf(grid)




# class Distribution(object):
#     """
#     General class to represent a distribution
#     """
#     def __init__(
#             self, xbasis=None, wbasis=None, smooth_sigma=1.0):
#         self.thr = 1e-8
#         self.smooth_sigma = smooth_sigma
#         self.xbasis = xbasis
#         self.wbasis = wbasis
#         self.clr_eval = None

#     def init_from_pdf(self, pdf_grid, pdf_eval):
#         self.pdf_grid = pdf_grid
#         self.pdf_eval = pdf_eval

#         self.cdf_grid = pdf_grid
#         self.cdf_eval = cumtrapz(pdf_eval, self.pdf_grid, initial=0)

#         self._invert_cdf()

#     def init_from_cdf(self, cdf_grid, cdf_eval):
#         self.cdf_grid = cdf_grid
#         self.cdf_eval = cdf_eval
#         self._invert_cdf()

#         self.pdf_eval = (cdf_eval[1:] - cdf_eval[:-1]) / np.diff(self.cdf_grid)
#         self.pdf_grid = self.cdf_grid[1:]

#         if self.xbasis is not None:
#             self.pdf_coeffs = self.xbasis.get_spline_expansion(
#                 self.pdf_eval, self.pdf_grid)

#             self.cdf_coeffs = self.xbasis.get_spline_expansion(
#                 self.cdf_eval, self.cdf_grid)

#     def init_from_quantile(self, quantile_grid, quantile_eval):
#         self.quantile_grid = quantile_grid
#         self.quantile_eval = quantile_eval
    
#     def _invert_cdf(self):
#         keep = np.where(np.diff(self.cdf_eval) > 1e-20)
#         self.quantile_grid = self.cdf_eval[keep]
#         self.quantile_eval = self.cdf_grid[keep]

#     def _invert_quantile(self):
#         cdf = PchipInterpolator(self.quantile_eval, self.quantile_grid)
#         self.cdf_grid = np.linspace(
#             self.quantile_eval[0], self.quantile_eval[-1], 1000)
#         self.pdf_grid = self.cdf_grid
        
#         pdf_eval = gaussian_filter1d(
#             cdf.derivative()(self.pdf_grid), sigma=self.smooth_sigma)
#         self.pdf_eval = pdf_eval / simps(pdf_eval, self.cdf_grid)
#         self.cdf_eval = cumtrapz(self.pdf_eval, self.pdf_grid, initial=0)

#     def compute_spline_expansions(self):
#         if self.xbasis is not None:
#             if self.clr_eval is not None:
#                 self.clr_coeffs = self.xbasis.get_spline_expansion(
#                     self.clr_eval, self.clr_grid)

#         if self.wbasis is not None:
#             try:
#                 self.quantile_coeffs = self.wbasis.get_spline_expansion(
#                     self.quantile_eval, self.quantile_grid)
#             except Exception as e:
#                 from scipy.interpolate import PchipInterpolator
#                 interp = PchipInterpolator(self.quantile_grid, self.quantile_eval)
#                 self.quantile_grid = np.linspace(0, 1, 200)
#                 self.quantile_eval = interp(self.quantile_grid)
#                 self.quantile_coeffs = self.wbasis.get_spline_expansion(
#                     self.quantile_eval, self.quantile_grid)

