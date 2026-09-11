import unittest

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import MODEL
from run_generated_sparse_solver import compile_and_bind, jacobian, solve


class GeneratedSparseSolverTest(unittest.TestCase):
    def test_real_jacobian(self):
        function=compile_and_bind(); circuit=build(True,amplitude=.1,circuit_type=CompactCircuit,**MODEL)
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored: state=stored["state"]
        matrix,scale=jacobian(circuit,state,.625e-6); rhs=np.sin(np.arange(98)+.25)/scale
        actual=solve(function,matrix,rhs)
        self.assertLess(np.linalg.norm(matrix@actual-rhs)/np.linalg.norm(rhs),1e-12)


if __name__=="__main__": unittest.main()
