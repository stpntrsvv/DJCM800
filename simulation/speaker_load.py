"""Линейные электрические эквиваленты кабинета на 16-омном выходе.

Это модель импеданса, влияющего на оконечник и ООС, а не акустический IR.
"""
from __future__ import annotations

import math
import numpy as np


BRIT_4X12_UK = {
    "name": "brit_4x12_uk_approx",
    "re_ohm": 12.5,
    "le_h": 0.8e-3,
    "resonance_hz": 100.0,
    "motional_peak_ohm": 70.0,
    "motional_q": 4.0,
}


def resolved(parameters=BRIT_4X12_UK):
    """Вернуть параметры, включая эквивалентные Lm/Cm параллельной ветви."""
    p = dict(parameters)
    omega = 2*math.pi*p["resonance_hz"]
    resistance = p["motional_peak_ohm"]
    quality = p["motional_q"]
    p["lm_h"] = resistance/(quality*omega)
    p["cm_f"] = quality/(resistance*omega)
    return p


def add_speaker_load(circuit, parameters=BRIT_4X12_UK):
    """Добавить Re+Le последовательно с параллельной Rm/Lm/Cm ветвью."""
    p = resolved(parameters)
    circuit.add("R", "Rspeaker_voice", "out", "speaker_voice", p["re_ohm"])
    circuit.add("L", "Lspeaker_voice", "speaker_voice", "speaker_motor", p["le_h"])
    circuit.add("R", "Rspeaker_motor", "speaker_motor", "0", p["motional_peak_ohm"])
    circuit.add("L", "Lspeaker_motor", "speaker_motor", "0", p["lm_h"])
    circuit.add("C", "Cspeaker_motor", "speaker_motor", "0", p["cm_f"])
    circuit.notes["speaker_load"] = (
        "Fractal X-Load UK / Marshall 4x12 Greenback approximation; "
        "100 Hz is documented, Re/Le/peak/Q are explicit engineering assumptions"
    )


def impedance(frequency_hz, parameters=BRIT_4X12_UK):
    """Комплексный входной импеданс эквивалента для проверки и графиков."""
    p = resolved(parameters)
    frequency = np.asarray(frequency_hz, dtype=float)
    omega = 2*np.pi*frequency
    admittance = 1/p["motional_peak_ohm"] + 1/(1j*omega*p["lm_h"]) + 1j*omega*p["cm_f"]
    return p["re_ohm"] + 1j*omega*p["le_h"] + 1/admittance
