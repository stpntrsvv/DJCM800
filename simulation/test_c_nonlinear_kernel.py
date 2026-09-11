import unittest

import numpy as np

from run_c_nonlinear_kernel import bind, call, compile_library, c_solve, reference


class CNonlinearKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path, _ = compile_library(); cls.library = bind(path)

    def check_model(self, name, inputs, function, jacobian_size):
        actual, jacobian = call(function, inputs, 3, jacobian_size, True)
        expected, expected_jacobian = reference(inputs, name)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(jacobian.reshape(expected_jacobian.shape), expected_jacobian,
                                   rtol=2e-12, atol=1e-15)

    def test_ecc83(self):
        self.check_model("ecc83", np.array([[20.,-5.],[150.,-1.],[300.,3.]]),
                         self.library.jcm800_dempwolf_rsd1_batch, 6)

    def test_el34(self):
        self.check_model("el34", np.array([[0.,-150.,250.],[50.,0.,400.],[800.,2.,500.]]),
                         self.library.jcm800_reefman_el34_batch, 9)

    def test_dense_solve(self):
        matrix=np.array([[4.,1.,2.],[1.,5.,1.],[2.,1.,6.]])
        rhs=np.array([1.,2.,3.])
        np.testing.assert_allclose(c_solve(self.library.jcm800_dense_solve,matrix,rhs),
                                   np.linalg.solve(matrix,rhs),rtol=2e-15,atol=1e-15)


if __name__ == "__main__": unittest.main()
