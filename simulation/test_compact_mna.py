"""Validate sparse stamps and the compact hot path independently of trajectories."""
import unittest
import numpy as np
from full_mna import Circuit, build
from compact_mna import CompactCircuit
from test_reduced_mna import ExactEliminationTests


class CompactAssemblyTests(unittest.TestCase):
    def test_stamps_across_models_and_trial_points(self):
        rng = np.random.default_rng(800)
        for tube_set in ("koren", "detailed:RSD-1", "detailed:RSD-2", "detailed:EHX-1"):
            with self.subTest(tube_set=tube_set):
                c = build(True, tube_set=tube_set, circuit_type=CompactCircuit)
                for x in [np.zeros(c.size), *(rng.normal(0., 50., c.size) for _ in range(12))]:
                    full = Circuit.nonlinear(c, x)
                    local = c.nonlinear(x)
                    for a, b in zip(full, local):
                        np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-15)
                    current, jac, charge, cap = c.compact_nonlinear(x[c.retained], derivatives=False)
                    np.testing.assert_array_equal(current, local[0][c.retained])
                    np.testing.assert_array_equal(charge, local[2][c.retained])
                    self.assertIsNone(jac)
                    self.assertIsNone(cap)

    def test_steps_without_full_nonlinear_api(self):
        factory = ExactEliminationTests().circuit
        for has_diode in (False, True):
            full, compact = factory(Circuit, has_diode), factory(CompactCircuit, has_diode)
            def forbidden(*args):
                raise AssertionError("Full-size nonlinear API called in hot path")
            compact.nonlinear = forbidden
            a, b = np.zeros(full.size), np.zeros(full.size)
            t = 0.
            for h in (1e-4, 5e-5, 1e-4, 5e-5):
                t += h
                a, sa = full.step(a, t, h)
                b, sb = compact.step(b, t, h)
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
                self.assertEqual(sa["iterations"], sb["iterations"])

    def test_analytic_jacobian_and_charge_derivative(self):
        c = build(False, controls=.5, tube_set="detailed:RSD-1", circuit_type=CompactCircuit)
        x = c.dc()[0][c.retained]
        _, jac, _, cap = c.compact_nonlinear(x)
        # Probe directions so cathode and shared-electrode contributions matter.
        rng = np.random.default_rng(34)
        for _ in range(8):
            direction = rng.normal(size=len(x))
            eps = 1e-4
            ip, _, qp, _ = c.compact_nonlinear(x+eps*direction, derivatives=False)
            im, _, qm, _ = c.compact_nonlinear(x-eps*direction, derivatives=False)
            np.testing.assert_allclose((ip-im)/(2*eps), jac@direction, rtol=2e-5, atol=1e-10)
            np.testing.assert_allclose((qp-qm)/(2*eps), cap@direction, rtol=2e-5, atol=1e-16)

    def test_recompile_rebuilds_local_stamps(self):
        c = build(False, tube_set="detailed:RSD-1", circuit_type=CompactCircuit)
        c.add("D", "D_extra", "new_node", "0", 1e-9, 1e-12, 1e-9)
        c.add("R", "R_extra", "new_node", "in", 1000.)
        c.compile()
        x = np.full(c.size, .15)
        for a, b in zip(Circuit.nonlinear(c, x), c.nonlinear(x)):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
