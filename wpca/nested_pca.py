import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.util import sum_product, quicksum

from wpca.base_pca import PCA
from wpca.spline import MonotoneQuadraticSplineBasis
from wpca.distribution import PeriodicDistribution
from wpca.transport import PeriodicTransportMap


class NestedPCA(PCA):
    def __init__(self, nbasis=None, spline_basis=None, compute_spline=True):

        super().__init__(nbasis, spline_basis, None,
                         None, compute_spline)

    def fit(self, measures, base_point, k):
        self.k = k
        self._initialize()
        self.base_point = base_point
        
        self.ndata = len(measures)
        spline_coeffs = []
        for m in measures:
            t = PeriodicTransportMap(base_point)
            t.compute(m)
            t.compute_spline_expansion(self.spline_basis)
            spline_coeffs.append(t.spline_coeffs)
        
        base_point.compute_quantile_spline_expansion(self.spline_basis)
        self.coeff_mat = np.stack(spline_coeffs)
        if self.cons_rhs is None:
            self.cons_rhs = np.diff(base_point.quantile_coeffs)

        self.center = np.mean(self.coeff_mat, axis=0)
        coeffs_centered = self.coeff_mat - self.center

        aux_ = np.dot(coeffs_centered.T, coeffs_centered)
        M_ = np.matmul(aux_ + np.eye(self.nbasis) * 1e-4, self.metric_aug)

        (eig_val, eig_vecs) = np.linalg.eig(M_)
        eig_val = np.real(eig_val)
        eig_vecs = np.real(eig_vecs)

        eig_vecs = eig_vecs / \
            np.sqrt(
                np.diag(np.matmul(np.matmul(eig_vecs.T, self.metric_aug), eig_vecs)))

        aux = np.argsort(eig_val)
        self.eig_val = np.flip(np.sort(eig_val))
        self.initialization = np.flip(eig_vecs[:, aux], axis=1)

        self.eig_vecs = np.zeros((self.nbasis, self.nbasis))
        self.coords = np.ones((self.ndata, self.nbasis))
        for i in range(self.k):
            self.eig_vecs[:, i], self.coords[:, i] = \
                self.find_component(i, coeffs_centered)


        self._finalize()

    def find_component2(self, n, coeffs_centered):
        def norm(model):
            out = 0
            for i in range(self.ndata):
                pt_i = np.copy(self.coeff_mat[i, :])
                center = np.copy(self.center)
                PT = []
                for s in range(self.nbasis):
                    vector = model.lamb[i] * model.w[s]
                    PT.append(center[s] + vector)

                for h in range(model.nvar):
                    for s in range(model.nvar):
                        out += (PT[h] - pt_i[h]) * \
                            self.metric_aug[h, s]*(PT[s] - pt_i[s])

            return out

        model = pyo.ConcreteModel()
        model.nvar = self.nbasis
        model.npoints = self.ndata

        model.w = pyo.Var(np.arange(model.nvar), domain=pyo.Reals,
                          initialize=lambda m, i: self.initialization[i, n])
        model.lamb = pyo.Var(np.arange(model.npoints),
                             domain=pyo.Reals, initialize=1)

        model.obj = pyo.Objective(rule=norm, sense=pyo.minimize)
        model.costr = pyo.ConstraintList()





    def find_component(self, n, coeffs_centered):
        def norm(model):
            out = 0
            for i in range(self.ndata):
                pt_i = np.copy(self.coeff_mat[i, :])
                center = np.copy(self.center)
                PT = []
                for s in range(self.nbasis):
                    vector = model.lamb[i] * model.w[s]
                    PT.append(center[s] + vector)

                for h in range(model.nvar):
                    for s in range(model.nvar):
                        out += (PT[h] - pt_i[h]) * \
                            self.metric_aug[h, s]*(PT[s] - pt_i[s])

            return out

        model = pyo.ConcreteModel()
        model.nvar = self.nbasis
        model.npoints = self.ndata

        model.w = pyo.Var(np.arange(model.nvar), domain=pyo.Reals,
                          initialize=lambda m, i: self.initialization[i, n])
        model.lamb = pyo.Var(np.arange(model.npoints),
                             domain=pyo.Reals, initialize=1)

        model.obj = pyo.Objective(rule=norm, sense=pyo.minimize)
        model.costr = pyo.ConstraintList()

        # costraint ||w||_E=1
        aux = 0
        for i in range(model.nvar):
            aux += self.metric_aug[i, i]*model.w[i]**2
            for j in range(i):
                aux += 2*model.w[i]*model.w[j]*self.metric_aug[i, j]

        model.costr.add(aux == 1)

        # monothonicity contstraint
        for i in range(self.ndata):
            x_i = np.copy(coeffs_centered[i, :])
            center = np.copy(self.center)
            for s in range(n):
                eig = np.copy(self.eig_vecs[:, s])
                center += eig * self.coords[i, s]

            for j in range(self.nbasis-1):
                costr = sum(
                    [self.constraints_mat[j, k] * (
                        model.lamb[i] * model.w[k] + center[k]) 
                     for k in range(self.nbasis)])
                model.costr.add(costr <= 0)

        # orthogonality constraints wrt previous components
        if n > 0:
            for s in range(n):
                eig = np.copy(self.eig_vecs[:, s])
                angle = 0
                for h in range(model.nvar):
                    for k in range(model.nvar):
                        angle += model.w[h]*eig[k]*self.metric_aug[h, k]
                model.costr.add(angle == 0)

        # # constraints on range of the function
        # model.costr.add(model.w[self.nbasis-1] - model.w[0] <= 1)

        # for i in range(self.ndata):
        #     center = np.copy(self.center)
        #     for s in range(n):
        #         eig = np.copy(self.eig_vecs[:, s])
        #         center += eig * self.coords[i, s]

        #     f_eval = self.spline_basis.eval_spline(center)
        #     for j in range(len(f_eval)):
        #         val = 0
        #         for k in range(self.nbasis):
        #             val += model.w[k] * model.lamb[i] * self.spline_basis.B[k, j] 
        #         # print(val)
        #         val = val + f_eval[j]
        #         try:
        #             model.costr.add(abs(val - self.spline_basis.xgrid[j]) <= 0)
        #         except Exception as e:
        #             print("skipping val: ", val, " in position ", j)

        solver = pyo.SolverFactory('ipopt')
        solver.solve(model)

        # needed to get argmin
        # TODO maybe find more intellingent way
        w_eval = np.ones((model.nvar,))
        for key, val in model.w.extract_values().items():
            w_eval[key] = val

        lamb_eval = np.ones((model.npoints,))
        for key, val in model.lamb.extract_values().items():
            lamb_eval[key] = val

        return w_eval, lamb_eval


    def transform(self, measures):
        if self.compute_spline:
            spline_coeffs = []
            for m in measures:
                t = PeriodicTransportMap(self.base_point)
                t.compute(m)
                t.compute_spline_expansion(self.spline_basis)
                spline_coeffs.append(t.spline_coeffs)

        coeff_mat = np.stack(spline_coeffs)
        self._transform_coeff_mat = coeff_mat
        coords = []
        for npt in range(len(measures)):
            pt = coeff_mat[npt, :] - self.center
            coord = [self.inner_prod(pt, self.eig_vecs[:, k])
                     for k in range(self.k)]
            coords.append(np.array(coord))

        X_trans = np.vstack(coords)
        # X_proj = np.zeros_like(X_trans)

        # self.X_trans = X_trans
        # for i in range(X_trans.shape[0]):
        #     pt = X_trans[i, :]
        #     pt_proj = self.project_on_sub(pt)
        #     X_proj[i, :] = pt_proj

        return X_trans
