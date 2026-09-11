import unittest,numpy as np
from compact_mna import CompactCircuit
from full_mna import ROOT,build
from run_controls_qualification import BASELINE,MODEL
from run_c_model98 import bind,nonlinear

class CModel98Test(unittest.TestCase):
    def test_nonlinear_at_accepted_state(self):
        function,rows,cols,g,c=bind(); circuit=build(True,amplitude=0.,controls=BASELINE,circuit_type=CompactCircuit,**MODEL)
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as z: state=z["state"]
        ci,cj,cq,cc=circuit.nonlinear(state);ai,aq,aj,ac=nonlinear(function,state)
        np.testing.assert_allclose(ai,ci,rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(aq,cq,rtol=2e-12,atol=1e-18)
        np.testing.assert_allclose(aj,cj[rows,cols],rtol=2e-11,atol=1e-15)
        np.testing.assert_allclose(ac,cc[rows,cols],rtol=2e-11,atol=1e-18)

if __name__=="__main__":unittest.main()
