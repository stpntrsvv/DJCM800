"""Полная float64 MNA с C sparse98 только для Newton linear solve."""
import numpy as np

from full_mna import Circuit
from run_generated_sparse_solver import compile_and_bind, solve


class CSparseCircuit(Circuit):
    _solver = None

    def compile(self):
        super().compile()
        self.factorizations = 0
        if self.size != 98:
            raise ValueError(f"generated sparse solver requires 98 unknowns, got {self.size}")
        if CSparseCircuit._solver is None:
            CSparseCircuit._solver = compile_and_bind()
        return self

    def linearized_delta(self, A, nonlinear_jacobian, residual, alpha):
        matrix=A+nonlinear_jacobian
        row=np.maximum(np.max(np.abs(matrix),axis=1),1e-15)
        return solve(self._solver,matrix/row[:,None],-residual/row)
