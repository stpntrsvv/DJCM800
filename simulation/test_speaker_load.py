import unittest

import numpy as np

from compact_mna import CompactCircuit
from full_mna import build
from speaker_load import BRIT_4X12_UK, impedance


class SpeakerLoadTest(unittest.TestCase):
    def test_resonance_and_asymptotes(self):
        z = impedance(np.array([1e-3, 100., 10_000.]))
        self.assertAlmostEqual(z[0].real, BRIT_4X12_UK["re_ohm"], places=3)
        self.assertAlmostEqual(abs(z[1]), 82.5, delta=0.1)
        self.assertGreater(abs(z[1]), 5*abs(z[0]))
        self.assertGreater(abs(z[2]), abs(z[0]))

    def test_load_is_part_of_compiled_mna(self):
        circuit = build(circuit_type=CompactCircuit, speaker_load=BRIT_4X12_UK)
        self.assertIn("speaker_motor", circuit.index)
        self.assertIn("Lspeaker_motor", circuit.branches)
        self.assertNotIn("Rload", [part[1] for part in circuit.parts])


if __name__ == "__main__":
    unittest.main()
