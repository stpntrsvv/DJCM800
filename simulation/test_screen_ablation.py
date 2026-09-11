import unittest

from compact_mna import CompactCircuit
from full_mna import build


class ScreenAblationTest(unittest.TestCase):
    def test_default_identity_is_preserved(self):
        a = build(circuit_type=CompactCircuit)
        b = build(circuit_type=CompactCircuit, el34_secondary_scale=1.)
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_secondary_and_component_overrides_are_explicit(self):
        circuit = build(circuit_type=CompactCircuit, tube_set="detailed:RSD-1", el34_secondary_scale=0.,
                        component_overrides={"Rch": 500.}, parameter_overrides={"RAA": 2200.})
        self.assertTrue(any(part[1] == "Xv4" and part[2] == "reefman-secondary:0" for part in circuit.parts))
        self.assertTrue(any(part[1] == "Rch" and part[4] == 500. for part in circuit.parts))
        baseline = build(True, circuit_type=CompactCircuit, tube_set="detailed:RSD-1")
        changed = build(True, circuit_type=CompactCircuit, tube_set="detailed:RSD-1",
                        parameter_overrides={"RAA": 2200.})
        base_l = next(part[4] for part in baseline.parts if part[1] == "Lsec16")
        changed_l = next(part[4] for part in changed.parts if part[1] == "Lsec16")
        self.assertAlmostEqual(base_l/changed_l, 2200/1700)


if __name__ == "__main__": unittest.main()
