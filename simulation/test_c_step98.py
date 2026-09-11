import math,unittest,numpy as np
from full_mna import ROOT,build
from run_controls_qualification import BASELINE,MODEL
from run_c_step98 import bind,step
class CStep98Test(unittest.TestCase):
    def test_one_real_step(self):
        f=bind();c=build(True,amplitude=.1,controls=BASELINE,**MODEL)
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as z:x,t=z["state"],float(z["time"])
        h=1.25e-6;py,_=c.step(x,t+h,h);native,_=step(f,x,t+h,h,.1*math.sin(2*math.pi*1000*(t+h)))
        np.testing.assert_allclose(native,py,rtol=2e-10,atol=1e-8)
if __name__=="__main__":unittest.main()
