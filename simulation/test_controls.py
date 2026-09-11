import unittest

from full_mna import build


class ControlBuildTests(unittest.TestCase):
    def test_mapping_changes_independent_controls(self):
        a = build(controls={"GAIN": .2, "MASTER": .8, "TREBLE": .1, "NFB": .5})
        b = build(controls={"GAIN": .8, "MASTER": .2, "TREBLE": .9, "NFB": 1.})
        self.assertNotEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.size, b.size)

    def test_unknown_control_is_rejected(self):
        with self.assertRaises(ValueError):
            build(controls={"VOLUME": .5})

    def test_time_function_overrides_source(self):
        circuit = build(amplitude=.123)
        circuit.source_function("Vin", lambda time: 2*time)
        self.assertEqual(circuit.rhs(.25)[circuit.branches["Vin"]], .5)
        self.assertEqual(circuit.rhs()[circuit.branches["Vin"]], 0.)


if __name__ == "__main__":
    unittest.main()
