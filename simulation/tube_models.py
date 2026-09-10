"""Отдельные float64-кандидаты; основной full_mna пока использует прежний закон.

Токи направлены в лампу. Напряжения относительно катода, единицы В/А.
Якобиан возвращает производные токов по (Va, Vg1[, Vg2]).
Источники и диапазоны: docs/11 Отдельная проверка ламп.md.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, wrightomega


@dataclass(frozen=True)
class TriodeParameters:
    G: float
    mu: float
    gamma: float
    C: float
    Gg: float
    xi: float
    Cg: float
    Ig0: float


# Dempwolf–Zölzer 2011, table 1. Measured specimens, not Philips nominal fits.
TRIODES = {
    "RSD-1": TriodeParameters(2.242e-3, 103.2, 1.26, 3.40, 6.177e-4, 1.314, 9.901, 8.025e-8),
    "RSD-2": TriodeParameters(2.173e-3, 100.2, 1.28, 3.19, 5.911e-4, 1.358, 11.76, 4.527e-8),
    "EHX-1": TriodeParameters(1.371e-3, 86.9, 1.349, 4.56, 3.263e-4, 1.156, 11.99, 3.917e-8),
}


def dempwolf(va, vg, parameters=TRIODES["RSD-1"]):
    """Equations 10–12, without clipping negative Ia or adding an invented knee.

    Published characterization: Va=20..300 V, Vg=-5..3 V. Evaluation elsewhere
    is diagnostic extrapolation. In particular, low-Va positive-grid is invalid.
    Returns (Ia, Ig1, Ik), followed by the 3x2 analytic Jacobian.
    """
    p = parameters
    x = va/p.mu + vg
    f = np.logaddexp(0., p.C*x)/p.C
    fg = np.logaddexp(0., p.Cg*vg)/p.Cg
    ik = p.G*f**p.gamma
    ig = p.Gg*fg**p.xi + p.Ig0
    dk = p.G*p.gamma*f**(p.gamma-1)*expit(p.C*x)
    dg = p.Gg*p.xi*fg**(p.xi-1)*expit(p.Cg*vg)
    return np.array([ik-ig, ig, ik]), np.array([[dk/p.mu, dk-dg], [0., dg], [dk/p.mu, dk]])


@dataclass(frozen=True)
class PentodeParameters:
    mu: float = 12.50
    exponent: float = 1.363
    kg1: float = 217.7
    kp: float = 50.5
    kvb: float = 1282.7
    kg2: float = 1950.2
    secondary: float = .022
    ap: float = .033
    w: float = 64.
    nu: float = 2.91
    lam: float = 4.23
    # Keep the author's rounded composite coefficients, do not silently
    # recompute them from independently rounded kg1/kg2.
    offset: float = .00408
    slope: float = 1.6e-6
    knee: float = .00408
    beta: float = .105
    screen_knee: float = 6.09


EL34 = PentodeParameters()


def reefman(va, vg, vs, parameters=EL34):
    """Reefman rational pentode + secondary emission, author EL34 coefficients.

    TubeLib.inc calls this BTetrodeD; its rational knee implements the pentode
    equations, not the exponential beam-tetrode knee BTetrodeDE.
    Ig1 is NOT part of these equations. Returns (Ia, Ig2, Ia+Ig2), J (3x3).
    Reject negative-anode/nonpositive-screen inputs: no fabricated continuation.
    Positive Vg1 is allowed only to expose extrapolation, not validated AB2.
    """
    if not np.isfinite([va, vg, vs]).all() or va < 0 or vs <= 0:
        raise ValueError("Reefman diagnostic domain requires finite Va>=0 and Vg2>0")
    p = parameters
    root = np.sqrt(p.kvb+vs*vs)
    z = p.kp*(1/p.mu+vg/root)
    f = np.logaddexp(0., z)
    e = vs*f/p.kp
    ip = e**p.exponent
    de = np.array([0., vs*expit(z)/root, f/p.kp-vs*vs*vg*expit(z)/root**3])
    dip = p.exponent*e**(p.exponent-1)*de
    inv = 1/(1+p.beta*va)
    dinv = np.array([-p.beta*inv*inv, 0., 0.])
    vco = vs/p.lam-p.nu*vg-p.w
    t = np.tanh(-p.ap*(va-vco))
    sec = p.secondary*va*(1+t)/p.kg2
    dsec = p.secondary/p.kg2*(np.array([1+t, 0., 0.]) + va*(1-t*t)*(-p.ap)*np.array([1., p.nu, -1/p.lam]))
    fa = p.offset+p.slope*va-p.knee*inv-sec
    fs = (1+p.screen_knee*inv)/p.kg2+sec
    dfa = np.array([p.slope, 0., 0.])-p.knee*dinv-dsec
    dfs = p.screen_knee/p.kg2*dinv+dsec
    ia, ig2 = ip*fa, ip*fs
    ja, js = dip*fa+ip*dfa, dip*fs+ip*dfs
    return np.array([ia, ig2, ia+ig2]), np.array([ja, js, ja+js])


def grid_diode(vg, resistance=2001., isat=1e-9, vt=8.617087e-5*300.15):
    """Exact DC solution of the OLD series-R/Shockley grid branch.

    This is retained only as a comparator, not a measured vacuum-grid model.
    Wright omega avoids overflow in Lambert W(exp(...)).
    """
    a = resistance*isat/vt
    omega = wrightomega(np.log(a)+(vg+resistance*isat)/vt)
    current = vt/resistance*omega-isat
    conductance = omega/(resistance*(1+omega))
    return float(current), float(conductance)
