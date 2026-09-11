"""Structural edge cases for exact elimination (run with unittest)."""
import unittest
import numpy as np
from full_mna import Circuit
from reduced_mna import ReducedCircuit


class ExactEliminationTests(unittest.TestCase):
    def circuit(self, cls, diode=False):
        c = cls()
        c.source("V1", "in", "0", dc=.5)
        c.add("R", "R1", "in", "out", 1000.)
        c.add("C", "C1", "out", "0", 1e-6)
        if diode:
            c.add("D", "D1", "out", "0", 1e-9, 1e-12, 1e-9)
        else:
            c.add("R", "R2", "out", "0", 2000.)
        return c.compile()

    def test_linear_and_diode_history_with_step_changes(self):
        for diode in (False, True):
            with self.subTest(diode=diode):
                full = self.circuit(Circuit, diode)
                reduced = self.circuit(ReducedCircuit, diode)
                # Start away from DC so capacitor and diode-charge histories matter.
                a, b = np.zeros(full.size), np.zeros(full.size)
                t = 0.
                for h in (1e-4, 5e-5, 1e-4, 5e-5):
                    t += h
                    a = full.step(a, t, h)[0]
                    b = reduced.step(b, t, h)[0]
                    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
                self.assertEqual(reduced.factorizations, 2)
                self.assertEqual(len(reduced.retained), int(diode))

    def test_no_linear_unknowns(self):
        circuits = []
        for cls in (Circuit, ReducedCircuit):
            c = cls()
            c.add("R", "R1", "out", "0", 1000.)
            c.add("D", "D1", "out", "0", 1e-9, 1e-12, 1e-9)
            circuits.append(c.compile())
        self.assertEqual(len(circuits[1].eliminated), 0)
        deltas = []
        for c in circuits:
            x = np.array([.2])
            current, jac, _, _ = c.nonlinear(x)
            deltas.append(c.linearized_delta(c.G, jac, c.G@x+current-np.array([.001]), 0.))
        np.testing.assert_array_equal(*deltas)

    def test_recompile_invalidates_linear_factorization(self):
        c = self.circuit(ReducedCircuit, True)
        c.dc()
        self.assertTrue(c._schur_cache)
        c.parts = [(*part[:4], 3000.) if part[1] == "R1" else part for part in c.parts]
        c.compile()
        self.assertFalse(c._schur_cache)
        full = self.circuit(Circuit, True)
        full.parts = list(c.parts)
        full.compile()
        np.testing.assert_allclose(c.dc()[0], full.dc()[0], rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
