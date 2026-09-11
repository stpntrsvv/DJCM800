import unittest

from full_mna import build, control_position


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

    def test_logarithmic_front_panel_positions(self):
        self.assertAlmostEqual(control_position("GAIN", 5.), .1)
        self.assertAlmostEqual(control_position("MASTER", 5.), .1)
        self.assertAlmostEqual(control_position("TREBLE", 5.), .5)
        self.assertEqual(control_position("GAIN", 0.), 0.)
        self.assertEqual(control_position("GAIN", 10.), 1.)

    def test_invalid_front_panel_position_is_rejected(self):
        with self.assertRaises(ValueError):
            control_position("GAIN", 11.)


if __name__ == "__main__":
    unittest.main()
