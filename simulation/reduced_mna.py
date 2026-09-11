"""Exact Schur elimination of linear-only MNA unknowns, with full state recovery.

Only the linear solve inside Newton changes. Currents, charges, integration,
line search and full-system stopping tolerances are inherited from Circuit.
G/C and the nonlinear inventory are fixed between calls to compile().
"""
from collections import OrderedDict
import warnings

from full_mna import Circuit
import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve


class ReducedCircuit(Circuit):
    def compile(self):
        super().compile()
        active = np.zeros(self.size, dtype=bool)
        for kind, control, output, _ in self.nl:
            active |= np.any(np.atleast_2d(control) != 0, axis=0)
            # Diode output is an outer product; tube output has one row per current.
            active |= np.any(np.atleast_2d(output) != 0, axis=0)
        self.retained = np.flatnonzero(active)
        self.eliminated = np.flatnonzero(~active)
        self._schur_cache = OrderedDict()
        self.factorizations = 0
        return self

    def _blocks(self, A, alpha):
        if alpha in self._schur_cache:
            self._schur_cache.move_to_end(alpha)
            return self._schur_cache[alpha]
        r, e = self.retained, self.eliminated
        Arr, Are = A[np.ix_(r, r)], A[np.ix_(r, e)]
        Aer, Aee = A[np.ix_(e, r)], A[np.ix_(e, e)]
        row = np.maximum(np.max(np.abs(Aee), axis=1), 1e-15)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            factor = lu_factor(Aee/row[:, None], check_finite=True)
        transfer = lu_solve(factor, Aer/row[:, None], check_finite=False)
        blocks = (Arr-Are@transfer, Are, transfer, factor, row)
        self._schur_cache[alpha] = blocks
        # Step bisection may create several matrices. Bound storage even for
        # callers supplying a continuously varying step.
        if len(self._schur_cache) > 16:
            self._schur_cache.popitem(last=False)
        self.factorizations += 1
        return blocks

    def linearized_delta(self, A, nonlinear_jacobian, residual, alpha):
        r, e = self.retained, self.eliminated
        if not len(e):
            return super().linearized_delta(A, nonlinear_jacobian, residual, alpha)
        schur, Are, transfer, factor, row = self._blocks(A, alpha)
        linear_rhs = lu_solve(factor, -residual[e]/row, check_finite=False)
        delta = np.empty(self.size)
        if len(r):
            J = schur+nonlinear_jacobian[np.ix_(r, r)]
            rhs = -residual[r]-Are@linear_rhs
            scaling = np.maximum(np.max(np.abs(J), axis=1), 1e-15)
            delta[r] = np.linalg.solve(J/scaling[:, None], rhs/scaling)
        delta[e] = linear_rhs-transfer@delta[r]
        return delta
