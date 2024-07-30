import numpy as np
import ott

import jax.numpy as jnp

from typing import List
from jax.tree_util import register_pytree_node_class
from ott.geometry import PointCloud
from ott.core import discrete_barycenter
from sklearn.metrics import pairwise_distances


from wpca.distribution import PeriodicDistribution
from wpca.transport import PeriodicTransportMap


class ProcrustesBarycenter(object):
    def __init__(self, threshold=1e-4, maxiter=100):
        self.threshold = threshold
        self.maxiter = maxiter

    def compute(self, measures: List[PeriodicDistribution]):
        xgrid = np.linspace(0, 1, 500)
        bary = PeriodicDistribution(supp_grid=xgrid, cdf_eval=xgrid)
        for _ in range(self.maxiter):
            bary, dist = self.procrustes_iter(bary, measures)
            print(dist)
            if dist < self.threshold:
                break

        self.bary_measure = bary

    def procrustes_iter(self, curr_bary:PeriodicDistribution, 
                        measures: List[PeriodicDistribution]):
        transports = []
        for m in measures:
            t = PeriodicTransportMap(curr_bary)
            t.compute(m)
            transports.append(t)
            
        eval_grid = np.linspace(-1, 2, 1000)
        t_eval = np.zeros_like(eval_grid)
        for t in transports:
            t_eval += t(eval_grid)
            
        t_eval = t_eval / len(measures)
        
        bary_map = PeriodicTransportMap(curr_bary)
        new_bary = bary_map.get_pushforward_from_transport(eval_grid, t_eval)
        trans = PeriodicTransportMap(curr_bary)
        trans.compute(new_bary)
        
        return new_bary, trans.wdist(trans.theta, new_bary.supp_grid)



@register_pytree_node_class
class PeriodicDist(ott.geometry.costs.CostFn):
  def pairwise(self, x, y):
    return -jnp.min((x - y - jnp.arange(-10, 10))**2)
    #return - jnp.sum((x- y)**4)

  def norm(self, x):
      return jnp.zeros(x.shape[0])


class SinkhornBarycenter(object):
    def __init__(self, xgrid, threshold=1e-6, max_iter=100):
        self.xgrid = xgrid
        self.thr = threshold
        self.max_iter = max_iter
        self.geometry = PointCloud(xgrid.reshape(-1, 1), cost_fn=PeriodicDist())

    def compute(self, measures):
        pmfs = []
        for m in measures:
            cdf = m.cdf_eval
            pdf = np.concatenate([[0], np.diff(cdf)])
            pmfs.append(jnp.array(pdf / np.sum(pdf)))

        self.pmfs = np.stack(pmfs)

        self.geometry._epsilon._target = 1e-6
        self.barycenter_pmf = discrete_barycenter.discrete_barycenter(
            self.geometry, self.pmfs, threshold=self.thr, 
            max_iterations=self.max_iter, lse_mode=False, debiased=True).histogram
        

