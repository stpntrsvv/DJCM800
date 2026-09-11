"""Local nonlinear stamps in the 39-variable Schur space, full-state Newton.

Element laws are imported unchanged. No interpolation, cached currents, fewer
Newton iterations, changed integration or relaxed full-system residual tests.
"""
from full_mna import diode, electrode
from reduced_mna import ReducedCircuit
import numpy as np
from scipy.linalg import lu_solve


class CompactCircuit(ReducedCircuit):
    def compile(self):
        super().compile()
        r = self.retained
        self._rr = np.ix_(r, r)
        self._local_nl = []
        for kind, control, output, params in self.nl:
            local_control = control[..., r]
            local_output = output[..., r]
            support = np.flatnonzero(
                np.any(np.atleast_2d(local_control) != 0, axis=0)
                | np.any(np.atleast_2d(local_output) != 0, axis=0))
            u = local_control[..., support]
            v = local_output[:, support] if kind == "T" else np.outer(u, u)
            self._local_nl.append((kind, support, np.ix_(support, support), u, v, params))
        self._tolerance = np.r_[np.full(len(self.nodes), 1e-10), np.full(len(self.branches), 1e-7)]
        return self

    def compact_nonlinear(self, xr, derivatives=True):
        """Stamp only each element's 1–4 incident nodes inside retained space."""
        n = len(self.retained)
        current, charge = np.zeros(n), np.zeros(n)
        jac = np.zeros((n, n)) if derivatives else None
        cap = np.zeros((n, n)) if derivatives else None
        for kind, support, block, control, output, params in self._local_nl:
            voltage = control@xr[support]
            if kind == "T":
                currents, derivative = electrode(params, voltage)
                current[support] += output.T@currents
                if derivatives:
                    jac[block] += output.T@derivative@control
            else:
                cur, conductance, q, capacitance = diode(float(voltage), *params)
                current[support] += control*cur
                charge[support] += control*q
                if derivatives:
                    jac[block] += output*conductance
                    cap[block] += output*capacitance
        return current, jac, charge, cap

    def _expand_vector(self, value):
        full = np.zeros(self.size)
        full[self.retained] = value
        return full

    def nonlinear(self, x):
        """Full-shape compatibility API for diagnostics, not the hot solve path."""
        current, jac, charge, cap = self.compact_nonlinear(x[self.retained])
        full_jac, full_cap = np.zeros_like(self.G), np.zeros_like(self.G)
        full_jac[self._rr], full_cap[self._rr] = jac, cap
        return self._expand_vector(current), full_jac, self._expand_vector(charge), full_cap

    def compact_delta(self, A, nonlinear_jacobian, residual, alpha):
        r, e = self.retained, self.eliminated
        if not len(e):
            J = A+nonlinear_jacobian
            scaling = np.maximum(np.max(np.abs(J), axis=1), 1e-15)
            return np.linalg.solve(J/scaling[:, None], -residual/scaling)
        schur, Are, transfer, factor, row = self._blocks(A, alpha)
        linear_rhs = lu_solve(factor, -residual[e]/row, check_finite=False)
        delta = np.empty(self.size)
        if len(r):
            J = schur+nonlinear_jacobian
            rhs = -residual[r]-Are@linear_rhs
            scaling = np.maximum(np.max(np.abs(J), axis=1), 1e-15)
            delta[r] = np.linalg.solve(J/scaling[:, None], rhs/scaling)
        delta[e] = linear_rhs-transfer@delta[r]
        return delta

    def newton(self, initial, b, alpha=0., history=None, maxiter=100):
        x = initial.copy()
        A = self.G+alpha*self.C
        history = np.zeros(self.size) if history is None else history
        tolerance = self._tolerance

        def residual_at(state, current, charge):
            # Keep the reference's arithmetic order and check every original row.
            return A@state+self._expand_vector(current)+alpha*self._expand_vector(charge)+history-b

        for it in range(maxiter):
            cur, jac, charge, cap = self.compact_nonlinear(x[self.retained])
            residual = residual_at(x, cur, charge)
            norm = np.max(np.abs(residual)/tolerance)
            if norm <= 1:
                return x, dict(iterations=it, kcl=float(np.max(np.abs(residual[:len(self.nodes)]), initial=0.)),
                               voltage=float(np.max(np.abs(residual[len(self.nodes):]), initial=0.)))
            delta = self.compact_delta(A, jac+alpha*cap, residual, alpha)
            accepted = False
            for power in range(28):
                candidate = x+delta*(.5**power)
                if not np.all(np.isfinite(candidate)):
                    continue
                ic, _, qc, _ = self.compact_nonlinear(candidate[self.retained], derivatives=False)
                rc = residual_at(candidate, ic, qc)
                nc = np.max(np.abs(rc)/tolerance)
                if nc < norm or nc <= 1:
                    x, accepted = candidate, True
                    break
            if not accepted:
                raise RuntimeError(f"Newton line search failed, iteration={it}, residual/tol={norm:g}")
        raise RuntimeError(f"Newton did not converge, residual/tol={norm:g}")

    def step(self, previous, t, h):
        _, _, charge, _ = self.compact_nonlinear(previous[self.retained], derivatives=False)
        history = -(self.C@previous+self._expand_vector(charge))/h
        return self.newton(previous, self.rhs(t), 1/h, history)
