"""Unreduced float64 MNA: R/C, coupled L, sources, electrode currents and charges.

No cascade partition, cached nonlinearity, polynomial root or fixed Newton count.
The same element inventory can be exported to ngspice for independent solving.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import math
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = {"meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "g": 1e9}
VT = 8.617087e-5 * 300.15


def expit(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def number(value, params):
    value = value.strip("{}")
    value = re.sub(r"(?<![\w.])(\d+(?:\.\d*)?|\.\d+)(meg|[kmunpg])\b",
                   lambda m: str(float(m[1]) * SCALE[m[2].lower()]), value, flags=re.I)
    return float(eval(value, {"__builtins__": {}},
                      dict(params, max=max, min=min, pot=lambda x: max(.001, min(.999, x)))))


def electrode(kind, v):
    """Currents and analytic Jacobian, coordinates relative to cathode."""
    if kind.startswith("dempwolf:"):
        from tube_models import TRIODES, dempwolf
        name = kind.split(":", 1)[1]
        currents, jac = dempwolf(v[0], v[1], TRIODES[name])
        return currents[:2], jac[:2]
    if kind == "reefman":
        from tube_models import reefman
        # The isolated law intentionally rejects unsupported quadrants. The MNA
        # nevertheless needs a finite trial-point continuation during source
        # stepping/Newton. Constant projection is numerical scaffolding, not a
        # claim about reverse-anode or nonpositive-screen tube physics.
        va, vs = max(float(v[0]), 0.), max(float(v[2]), 1e-6)
        currents, jac = reefman(va, v[1], vs)
        if v[0] < 0:
            jac[:, 0] = 0.
        if v[2] <= 1e-6:
            jac[:, 2] = 0.
        return currents[:2], jac[:2]
    p, g = v[:2]
    if kind == "triode":
        root = math.sqrt(300 + p*p)
        z = 600 * (.01 + g/root)
        sp, sig = np.logaddexp(0., z), expit(z)
        e = p*sp/600
        if e <= 0:
            return np.array([0.]), np.zeros((1, 2))
        de_p = sp/600 - sig*g*p*p/root**3
        de_g = p*sig/root
        factor = 2*1.4*e**.4/1060
        return np.array([2*e**1.4/1060]), np.array([[factor*de_p, factor*de_g]])
    s = v[2]
    se = max(s, .001)
    z = 60*(1/11 + g/se)
    sp, sig = np.logaddexp(0., z), expit(z)
    e = max(s, 0)*sp/60
    de_g = max(s, 0)*sig/se
    de_s = (sp/60 if s > 0 else 0) - (s*sig*g/se**2 if s > .001 else 0)
    e = max(e, 0.)
    a = math.atan(max(p, 0)/24)
    scale = 2*e**1.35/650
    factor = 2*1.35*e**.35/650 if e else 0.
    ip = scale*a
    dip = scale/(24*(1+(p/24)**2)) if p > 0 else 0.
    u = max(s/11+g, 0.)
    screen = u**1.35/4200
    ds = 1.35*u**.35/4200 if u else 0.
    return np.array([ip, screen]), np.array([[dip, factor*de_g*a, factor*de_s*a], [0, ds, ds/11]])


def diode(v, isat, cjo, tt):
    # Linear extension above 40 thermal voltages protects Newton trial points.
    z = v/VT
    if z < 40:
        e = math.exp(z)
        cur, conductance = isat*math.expm1(z), isat*e/VT
    else:
        e = math.exp(40)
        cur, conductance = isat*(e*(1+z-40)-1), isat*e/VT
    # SPICE-like M=.5, Vj=1V, Fc=.5 junction-charge continuation.
    if v < .5:
        charge = 2*cjo*(1-math.sqrt(1-v))
        capacitance = cjo/math.sqrt(1-v)
    else:
        d = v-.5
        charge = 2*cjo*(1-math.sqrt(.5)) + cjo*math.sqrt(2)*(d+.5*d*d)
        capacitance = cjo*math.sqrt(2)*(1+d)
    return cur, conductance, charge+tt*cur, capacitance+tt*conductance


class Circuit:
    def __init__(self, tube_set="koren", tube_caps="auto", el34_grid_r=None):
        self.parts = []
        self.sources = {}
        self.notes = {}
        self.tube_set = tube_set
        self.tube_caps = tube_caps
        self.el34_grid_r = (1001. if tube_set == "koren" else 2001.) if el34_grid_r is None else el34_grid_r

    def add(self, kind, name, *args):
        self.parts.append((kind, name, *args))

    def source(self, name, p, n, dc=0., amplitude=0., frequency=0., phase=0.):
        self.add("V", name, p, n)
        self.sources[name] = (dc, amplitude, frequency, phase)

    def tube(self, name, p, g, k, screen=None):
        if self.tube_set == "koren":
            kind = "triode" if screen is None else "pentode"
        elif self.tube_set.startswith("detailed:"):
            triode_name = self.tube_set.split(":", 1)[1]
            if triode_name not in ("RSD-1", "RSD-2", "EHX-1"):
                raise ValueError(f"Unknown detailed triode parameter set: {triode_name}")
            kind = f"dempwolf:{triode_name}" if screen is None else "reefman"
        else:
            raise ValueError(f"Unknown tube set: {self.tube_set}")
        self.add("T", name, kind, p, g, k, screen)
        detailed_caps = self.tube_caps == "datasheet" or (self.tube_caps == "auto" and self.tube_set != "koren")
        if screen is None:
            c = (1.6e-12, 1.6e-12, .33e-12) if detailed_caps else (2.3e-12, 2.4e-12, .9e-12)
        else:
            c = (15.2e-12, 1.1e-12, 8.4e-12) if detailed_caps else (15e-12, 1e-12, 8e-12)
        for suffix, a, b, value in zip(("gk", "gp", "pk"), (g, g, p), (k, p, k), c):
            self.add("C", f"C_{name}_{suffix}", a, b, value)
        self.add("R", f"R_{name}_pk", p, k, 1e9)
        j = name.lower()+"_junction"
        # Dempwolf already returns Ig1. Reefman does not, so its provisional
        # legacy R+Shockley grid branch remains explicit until EL34 Ig1 is qualified.
        if not kind.startswith("dempwolf:"):
            resistance = 2001. if screen is None else self.el34_grid_r
            self.add("R", f"R_{name}_grid", g, j, resistance)
            self.add("D", f"D_{name}_grid", j, k, 1e-9, 10e-12, 1e-9)

    def compile(self):
        nodes = []
        for kind, name, *args in self.parts:
            if kind == "K":
                continue
            terms = [args[1], args[2], args[3], args[4]] if kind == "T" else args[:4] if kind == "E" else args[:2]
            for node in terms:
                if node is not None and node != "0" and node not in nodes:
                    nodes.append(node)
        self.nodes = nodes
        self.index = {node: i for i, node in enumerate(nodes)}
        self.branches = {p[1]: len(nodes)+i for i, p in enumerate(p for p in self.parts if p[0] in ("L", "V", "E"))}
        self.size = len(nodes)+len(self.branches)
        self.G = np.zeros((self.size, self.size))
        self.C = np.zeros_like(self.G)
        self.nl = []
        inductors = {p[1]: p[4] for p in self.parts if p[0] == "L"}
        for kind, name, *args in self.parts:
            if kind == "K":
                names, coeff = args
                for i, a in enumerate(names):
                    for b in names[i+1:]:
                        ia, ib = self.branches[a], self.branches[b]
                        self.C[ia, ib] = self.C[ib, ia] = -coeff*math.sqrt(inductors[a]*inductors[b])
                continue
            if kind == "T":
                model, p, g, k, s = args
                control = np.array([self.inc(p, k), self.inc(g, k)] + ([] if s is None else [self.inc(s, k)]))
                output = control[[0, 1]] if model.startswith("dempwolf:") else control[[0]] if s is None else control[[0, 2]]
                self.nl.append(("T", control, output, model))
                continue
            inc = self.inc(args[0], args[1])
            if kind == "R":
                self.G += np.outer(inc, inc)/args[2]
            elif kind == "C":
                self.C += np.outer(inc, inc)*args[2]
            elif kind in ("L", "V", "E"):
                idx = self.branches[name]
                self.G[:, idx] += inc
                self.G[idx, :] += inc
                if kind == "L":
                    self.C[idx, idx] = -args[2]
                elif kind == "E":
                    self.G[idx, :] -= args[4]*self.inc(args[2], args[3])
            elif kind == "D":
                self.nl.append(("D", inc, np.outer(inc, inc), args[2:]))
            else:
                raise ValueError(kind)
        return self

    def inc(self, p, n):
        v = np.zeros(self.size)
        if p != "0": v[self.index[p]] += 1
        if n != "0": v[self.index[n]] -= 1
        return v

    def rhs(self, t=None, scale=1.):
        b = np.zeros(self.size)
        for name, (dc, amp, freq, phase) in self.sources.items():
            b[self.branches[name]] = scale*(dc if t is None else dc+amp*math.sin(2*math.pi*freq*t+phase))
        return b

    def nonlinear(self, x):
        i, q = np.zeros(self.size), np.zeros(self.size)
        j, c = np.zeros_like(self.G), np.zeros_like(self.G)
        for kind, control, output, params in self.nl:
            if kind == "T":
                currents, jac = electrode(params, control@x)
                i += output.T@currents
                j += output.T@jac@control
            else:
                cur, cond, charge, cap = diode(float(control@x), *params)
                i += control*cur
                j += output*cond
                q += control*charge
                c += output*cap
        return i, j, q, c

    def linearized_delta(self, A, nonlinear_jacobian, residual, alpha):
        """Solve one Newton correction; overridden by exact linear elimination."""
        J = A+nonlinear_jacobian
        row = np.maximum(np.max(np.abs(J), axis=1), 1e-15)
        return np.linalg.solve(J/row[:, None], -residual/row)

    def newton(self, initial, b, alpha=0., history=None, maxiter=100):
        x = initial.copy()
        A = self.G+alpha*self.C
        history = np.zeros(self.size) if history is None else history
        # KCL rows are amperes; source/inductor equations are volts.
        tolerance = np.r_[np.full(len(self.nodes), 1e-10), np.full(len(self.branches), 1e-7)]
        for it in range(maxiter):
            cur, jac, charge, cap = self.nonlinear(x)
            residual = A@x+cur+alpha*charge+history-b
            norm = np.max(np.abs(residual)/tolerance)
            if norm <= 1:
                return x, dict(iterations=it, kcl=float(np.max(np.abs(residual[:len(self.nodes)]))),
                               voltage=float(np.max(np.abs(residual[len(self.nodes):]))))
            delta = self.linearized_delta(A, jac+alpha*cap, residual, alpha)
            accepted = False
            for power in range(28):
                candidate = x+delta*(.5**power)
                if not np.all(np.isfinite(candidate)):
                    continue
                ic, _, qc, _ = self.nonlinear(candidate)
                rc = A@candidate+ic+alpha*qc+history-b
                nc = np.max(np.abs(rc)/tolerance)
                if nc < norm or nc <= 1:
                    x, accepted = candidate, True
                    break
            if not accepted:
                raise RuntimeError(f"Newton line search failed, iteration={it}, residual/tol={norm:g}")
        raise RuntimeError(f"Newton did not converge, residual/tol={norm:g}")

    def dc(self):
        x = np.zeros(self.size)
        # Source continuation changes the DC problem only during initialization.
        # Final returned solution must satisfy full-source original equations.
        reports = []
        for scale in np.linspace(0, 1, 101):
            x, stats = self.newton(x, self.rhs(scale=scale))
            reports.append(stats)
        return x, reports

    def step(self, previous, t, h):
        _, _, q, _ = self.nonlinear(previous)
        history = -(self.C@previous+q)/h
        return self.newton(previous, self.rhs(t), 1/h, history)

    def export(self):
        lines = ["* Unreduced circuit exported from full_mna.py, explicit electrode laws",
                 f".param VT={VT:.17g}",
                 ".func sp(x) {max(x,0)+ln(1+exp(-abs(x)))}"]
        for kind, name, *args in self.parts:
            if kind in ("R", "C", "L"):
                lines.append(f"{name} {args[0]} {args[1]} {args[2]:.17g}")
            elif kind == "K":
                lines.append(f"{name} {' '.join(args[0])} {args[1]:.17g}")
            elif kind == "V":
                dc, amp, freq, phase = self.sources[name]
                # All present AC sources use zero DC offset in this experiment.
                lines.append(f"{name} {args[0]} {args[1]} DC {dc:.17g} SIN({dc:.17g} {amp:.17g} {freq:.17g} 0 0 {phase*180/math.pi:.17g})")
            elif kind == "E":
                lines.append(f"{name} {' '.join(args[:4])} {args[4]:.17g}")
            elif kind == "D":
                p, n, iss, cjo, tt = args
                lines.append(f"{name} {p} {n} M_{name}")
                lines.append(f".model M_{name} D(Is={iss:.17g} N=1 Rs=0 Cjo={cjo:.17g} Vj=1 M=0.5 Fc=0.5 Tt={tt:.17g})")
            elif kind == "T":
                model, p, g, k, s = args
                vp, vg = f"v({p},{k})", f"v({g},{k})"
                if model == "triode":
                    e = f"({vp}/600*sp(600*(0.01+{vg}/sqrt(300+{vp}^2))))"
                    ip = f"2*max({e},0)^1.4/1060"
                elif model == "pentode":
                    vs = f"v({s},{k})"
                    e = f"(max({vs},0)/60*sp(60*(1/11+{vg}/max({vs},0.001))))"
                    ip = f"2*max({e},0)^1.35/650*atan(max({vp},0)/24)"
                    lines.append(f"B_{name}_screen {s} {k} I={{max({vs}/11+{vg},0)^1.35/4200}}")
                else:
                    raise ValueError(f"SPICE export is not implemented for tube model {model}")
                lines.append(f"B_{name}_plate {p} {k} I={{{ip}}}")
        return "\n".join(lines)+"\n"

    def fingerprint(self):
        """Stable identity also available for models not exported to SPICE."""
        return repr((self.tube_set, self.tube_caps, self.el34_grid_r,
                     self.parts, sorted(self.sources.items())))


def build(full_supply=False, amplitude=.001, controls=.1, tube_set="koren",
          tube_caps="auto", el34_grid_r=None, circuit_type=Circuit):
    circuit = circuit_type(tube_set=tube_set, tube_caps=tube_caps, el34_grid_r=el34_grid_r)
    params = {}
    for line in (ROOT / "simulation/ngspice/jcm800_2203_1981.inc").read_text(encoding="utf-8").splitlines():
        if line.startswith(".param"):
            for field in line.split()[1:]:
                key, value = field.split("=")
                params[key] = number(value, params)
    params.update(GAIN=controls, MASTER=controls)
    for line in (ROOT / "simulation/ngspice/jcm800_2203_1981.inc").read_text(encoding="utf-8").splitlines():
        if not line or line[0] in "*.": continue
        name, *a = line.split()
        kind = name[0].upper()
        if kind in "RCL": circuit.add(kind, name, a[0], a[1], number(a[2], params))
        elif kind == "V": circuit.source(name, a[0], a[1], dc=number(a[2], params))
        elif kind == "E": circuit.add(kind, name, *a[:4], number(a[4], params))
        elif kind == "K": circuit.add(kind, name, a[:-1], number(a[-1], params))
        elif kind == "X": circuit.tube(name, *a[:-1])
        else: raise ValueError(line)
    circuit.source("Vin", "in", "0", amplitude=amplitude, frequency=1000)
    if full_supply:
        extend_factory_supply(circuit)
    return circuit.compile()


def extend_factory_supply(c):
    remove = {"Vraw", "Rps", "Vbias", "Rmainbalance1", "Rmainbalance2", "Ls", "Kmain", "Rsec", "Etap"}
    c.parts = [p for p in c.parts if p[1] not in remove]
    for name in ("Vraw", "Vbias"): del c.sources[name]
    # Reservoir is before HT fuse; center tap is its capacitor midpoint.
    c.parts = [tuple([p[0], p[1], "reservoir", *p[3:]]) if p[1] == "Cmain1" else p for p in c.parts]
    c.source("Vmains", "line", "0", amplitude=230*math.sqrt(2), frequency=50)
    c.add("R", "Rmains_closed", "line", "primary", 5.)
    c.add("L", "Lprimary", "primary", "0", 40.)
    c.add("L", "Lhv1", "hv1i", "main_mid", 40*(175/230)**2)
    c.add("L", "Lhv2", "main_mid", "hv2i", 40*(175/230)**2)
    c.add("R", "Rhv1", "hv1i", "hv1", 15.)
    c.add("R", "Rhv2", "hv2i", "hv2", 15.)
    c.add("L", "Lbiasw", "bias_ac_i", "0", 40*(100/230)**2)
    c.add("R", "Rbiasw", "bias_ac_i", "bias_ac", 100.)
    c.add("L", "Lheater1", "heater1i", "0", 40*(3.15/230)**2)
    c.add("L", "Lheater2", "0", "heater2i", 40*(3.15/230)**2)
    c.add("R", "Rheater1", "heater1i", "heater1", .015)
    c.add("R", "Rheater2", "heater2i", "heater2", .015)
    c.add("R", "Rheater_load", "heater1", "heater2", 6.3/6.9)
    c.add("K", "Kpower", ["Lprimary", "Lhv1", "Lhv2", "Lbiasw", "Lheater1", "Lheater2"], .9999)
    for name, p, n in (("Dbridge1", "hv1", "reservoir"), ("Dbridge2", "hv2", "reservoir"),
                       ("Dbridge3", "0", "hv1"), ("Dbridge4", "0", "hv2")):
        c.add("R", "R_"+name, p, name+"_a", .05)
        c.add("D", name, name+"_a", n, 5e-9, 20e-12, 2e-6)
    c.add("R", "Rht_closed", "reservoir", "bplus", .01)
    c.add("R", "Rbias_feed", "bias_ac", "bias_d", 27e3)
    c.add("D", "Dbias", "bias_raw", "bias_d", 5e-9, 20e-12, 2e-6)
    c.add("C", "Cbias_raw", "bias_raw", "0", 10e-6)
    c.add("R", "Rbias_filter", "bias_raw", "bias", 15e3)
    c.add("C", "Cbias", "bias", "0", 10e-6)
    c.add("R", "Rbias_bottom", "bias", "bias_pot", 56e3)
    c.add("R", "Rbias_adjust", "bias_pot", "0", 11e3)
    # Explicit loaded 4/8/16-ohm winding sections instead of a sensed voltage tap.
    previous, turns = "0", 0.
    for z in (4, 8, 16):
        delta = math.sqrt(z)-turns
        end = "out" if z == 16 else f"tap{z}"
        c.add("L", f"Lsec{z}", f"sec{z}i", previous, 8.85*delta**2/1700)
        c.add("R", f"Rsec{z}", f"sec{z}i", end, .320*delta/4)
        previous, turns = end, math.sqrt(z)
    c.add("K", "Koutput", ["Lpa", "Lpb", "Lsec4", "Lsec8", "Lsec16"], .9995)
    c.notes["unmeasured"] = "PT winding L/R/k and ratios, choke L/R, OT original magnetic properties, rectifier parameters, heater thermal dynamics"
