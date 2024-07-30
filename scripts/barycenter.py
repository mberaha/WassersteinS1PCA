import argparse
import numpy as np
import os
import pickle
from scipy.stats import beta, norm, uniform
from scipy.integrate import trapz

from wpca.distribution import PeriodicDistribution
from wpca.transport import PeriodicTransportMap
from wpca.barycenter import ProcrustesBarycenter, SinkhornBarycenter



def run_one(measures, xgrid, name, base_path):
    wass_bary = ProcrustesBarycenter(1e-3, 50)
    sinkhorn_bary = SinkhornBarycenter(xgrid)
    wass_bary.compute(measures)
    sinkhorn_bary.compute(measures)

    bary_measure = wass_bary.bary_measure
    transports_from_bary = []
    for m in measures:
        t = PeriodicTransportMap(bary_measure)
        t.compute(m)
        transports_from_bary.append(t)

    n = len(measures)
    quant_grid = np.linspace(0, 1, 1000)

    bary_pdf = bary_measure.eval_smooth_pdf(quant_grid)

    w_dists = np.zeros((n, n))
    log_dists = np.zeros((n, n))
    for i in range(n):
        print("i: {0} / {1}".format(i+1, n))
        for j in range(n):
            tmap = PeriodicTransportMap(measures[i])
            tmap.compute(measures[j])
            w_dists[i, j] = tmap.wdist(tmap.theta, quant_grid)
            
            log_dists[i, j] = trapz(
                (transports_from_bary[i](quant_grid) - 
                 transports_from_bary[j](quant_grid))**2 * bary_pdf,
                quant_grid)

    filename = os.path.join(base_path, "{0}.pickle".format(name))
    with open(filename, "wb") as fp:
        pickle.dump({
                "measures": measures,
                "bary_w": wass_bary,
                "bary_s": sinkhorn_bary,
                "w_dists": w_dists,
                "log_dists": log_dists},
            fp
        )


def uniform_two_groups(xgrid):
    measures = []
    for i in range(10):
        if i < 5:
            c = 1/4
            w = 0.1 + 0.2 / 5 * i
        else:
            c= 3/4
            w = 0.1 + 0.2 / 5 * (i - 4)
        p_func = lambda x: uniform.cdf(x, c - w/2, w)
        dist = PeriodicDistribution(supp_grid=xgrid, cdf_eval=p_func(xgrid))
        measures.append(dist)
    
    return measures

def uniform_three_groups(xgrid):
    measures = []
    centers = [0, 1/3, 2/3]

    for i in range(10):
        w = 0.05 + 0.15 / 10 * i
        pdf = np.zeros_like(xgrid) 
        pdf[((xgrid <= w/2) | (xgrid >= 1 - w/2))] = 1 / w
        cdf = np.cumsum(pdf) * (xgrid[1] - xgrid[0])
        dist = PeriodicDistribution(supp_grid=xgrid, cdf_eval=cdf)
        measures.append(dist)
        

    for i in range(20):
        if i < 10:
            c = centers[1]
            w = 0.05 + 0.15 / 10 * i
        else:
            c = centers[2]
            w = 0.05 + 0.15 / 10 * (i - 10)
            
        p_func = lambda x: uniform.cdf(x, c - w/2, w)
        dist = PeriodicDistribution(supp_grid=xgrid, cdf_eval=p_func(xgrid))
        measures.append(dist)
    return measures

def beta_two_groups(xgrid):
    measures = []
    for i in range(20):
        b = 2
        if i < 10:
            a = np.random.uniform(1.1, 1.5)
        else:
            a = np.random.uniform(2.2, 3)
        pfunc = lambda x: beta.cdf(x, a, b)
        dist = PeriodicDistribution(supp_grid=xgrid, cdf_eval=pfunc(xgrid))
        measures.append(dist)
    return measures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data/bary_out")
    args = parser.parse_args()

    xgrid = np.linspace(0.0001, 0.9999, 1000)
    run_one(uniform_two_groups(xgrid), xgrid, "uniform_two_groups", 
            args.output_path)
    run_one(uniform_three_groups(xgrid), xgrid, "uniform_three_groups", 
            args.output_path)
    run_one(beta_two_groups(xgrid), xgrid, "beta_two_groups", 
            args.output_path)
