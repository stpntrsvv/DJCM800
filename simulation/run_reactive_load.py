"""Сравнить 16 Ом и реактивный Brit 4x12 на полном гитарном аккорде."""
from __future__ import annotations

import argparse
import hashlib
import json
import wave

import numpy as np

from compact_mna import CompactCircuit
from full_mna import ROOT, build
from run_controls_qualification import BASELINE, MODEL
from run_physics_hypotheses import diagnostics, simulate
from speaker_load import BRIT_4X12_UK, impedance, resolved

SAMPLE = ROOT/"simulation/samples/e_major_chord/e_major_dry.wav"
DEST = ROOT/"simulation/experiments/reactive_load"
RAW = ROOT/"simulation/raw/reactive_load"
POSITIONS = dict(GAIN=5., BASS=5., MID=5., TREBLE=5., MASTER=5., PRESENCE=5.)


def read_signal():
    digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    with wave.open(str(SAMPLE), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("expected mono PCM16 WAV")
        rate = stream.getframerate()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float)
    data -= np.mean(data)
    data /= np.max(np.abs(data))
    return data, rate, digest


def circuit(load):
    options = dict(full_supply=True, amplitude=0., controls=BASELINE,
                   control_positions=POSITIONS, circuit_type=CompactCircuit, **MODEL)
    if load == "reactive":
        options["speaker_load"] = BRIT_4X12_UK
    return build(**options)


def transfer_state(source, target, state):
    """Перенести общее физическое состояние; новые состояния нагрузки начать с нуля."""
    result = np.zeros(target.size)
    for name, index in target.index.items():
        if name in source.index:
            result[index] = state[source.index[name]]
    for name, index in target.branches.items():
        if name in source.branches:
            result[index] = state[source.branches[name]]
    return result


def load_metrics(c, states, h, kind):
    voltage = states[:, c.index["out"]]
    if kind == "resistive":
        current = voltage/16.
        real_power = voltage*current
    else:
        current = states[:, c.branches["Lspeaker_voice"]]
        motor = states[:, c.index["speaker_motor"]]
        p = resolved()
        real_power = current*current*p["re_ohm"] + motor*motor/p["motional_peak_ohm"]
    apparent = voltage*current
    first = min(len(states), round(.020/h)+1)
    tail = min(len(states), round(.500/h)+1)
    return dict(
        voltage_peak_v=float(np.max(np.abs(voltage))),
        voltage_rms_v=float(np.sqrt(np.mean(voltage*voltage))),
        voltage_first20_rms_v=float(np.sqrt(np.mean(voltage[:first]**2))),
        voltage_last500_rms_v=float(np.sqrt(np.mean(voltage[-tail:]**2))),
        current_peak_a=float(np.max(np.abs(current))),
        real_power_mean_w=float(np.mean(real_power)),
        real_power_first20_mean_w=float(np.mean(real_power[:first])),
        real_power_last500_mean_w=float(np.mean(real_power[-tail:])),
        real_energy_j=float(np.sum(real_power)*h),
        port_power_mean_w=float(np.mean(apparent)),
    )


def write_wav(path, signal, rate, scale):
    pcm = np.int16(np.clip(signal/scale, -1., 1.)*32767.)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())


def render(result):
    lines = ["# Реактивная нагрузка Brit 4×12", "", f"Статус: **{result['status']}**.", "",
             "Сравниваются резистор 16 Ом и включённый внутрь MNA электрический эквивалент",
             "закрытого Marshall-подобного 4×12. Это нагрузка оконечника и ООС, не кабинетный IR.", "",
             "Документированная привязка Fractal X-Load UK задаёт тип кабинета и резонанс около",
             "100 Гц. Re, Le, высота и Q резонанса источником не опубликованы и поэтому являются",
             "явными инженерными параметрами этого опыта.", "", "## Результаты", "",
             "| Нагрузка | Выход peak, В | P real avg, Вт | P первые 20 мс, Вт | P последние 500 мс, Вт | B+ min, В | Экран mean/max20, Вт | WAV |",
             "|:---|---:|---:|---:|---:|---:|---:|:---|"]
    for row in result.get("cases", []):
        d, load = row["amplifier"], row["load"]
        lines.append(f"| {row['name']} | {load['voltage_peak_v']:.3f} | {load['real_power_mean_w']:.3f} | "
                     f"{load['real_power_first20_mean_w']:.3f} | {load['real_power_last500_mean_w']:.3f} | {d['bplus_min_v']:.3f} | "
                     f"{d['el34_screen_mean_max_w']:.3f}/{d['el34_screen_max_20ms_w']:.3f} | "
                     f"[{row['name']}.wav]({row['name']}.wav) |")
    if result.get("comparison"):
        comparison = result["comparison"]
        lines += ["", f"RMS-разность выходных форм: **{comparison['output_relative_rms_difference']*100:.2f}%**; "
                  f"после оптимального скалярного выравнивания: **{comparison['output_shape_difference_after_gain']*100:.2f}%**."]
    lines += ["", "Параметры эквивалента и комплексная кривая 20 Гц–20 кГц сохранены в",
              "`metrics.json` и `impedance.csv`. WAV имеют одну общую шкалу; абсолютные",
              "электрические величины следует брать из JSON.", "",
              "Источник: https://wiki.fractalaudio.com/wiki/index.php?title=X-Load_LB-2_Reactive_Load_Box", ""]
    if "error" in result:
        lines += ["## Ошибка", "", result["error"], ""]
    (DEST/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global DEST, RAW
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-chord", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.run_name:
        if not all(ch.isalnum() or ch in "_-" for ch in args.run_name):
            parser.error("invalid run-name")
        DEST, RAW = DEST/args.run_name, RAW/args.run_name
    if args.quick:
        DEST, RAW = DEST/"quick", RAW/"quick"
    DEST.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    output = DEST/"metrics.json"
    if args.report_only:
        result = json.loads(output.read_text(encoding="utf-8"))
        arrays = {}
        for row in result["cases"]:
            c = circuit(row["name"])
            states = np.load(RAW/f"{row['name']}.npz")["state"]
            row["amplifier"] = diagnostics(c, states, result["h_s"], MODEL["el34_grid_r"])
            row["load"] = load_metrics(c, states, result["h_s"], row["name"])
            arrays[row["name"]] = states[:, c.index["out"]]
        reference, candidate = arrays["resistive"], arrays["reactive"]
        gain = float(reference@candidate/(candidate@candidate))
        result["comparison"] = dict(
            output_relative_rms_difference=float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference)),
            output_shape_difference_after_gain=float(np.linalg.norm(gain*candidate-reference)/np.linalg.norm(reference)),
            reactive_gain_to_resistive=gain)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        render(result); return
    if output.exists() and not args.quick:
        raise FileExistsError("use --run-name")
    signal, rate, digest = read_signal()
    duration = .002 if args.quick else (len(signal)/rate if args.full_chord else .020)
    h = 20e-6 if args.quick else (5e-6 if args.full_chord else 1.25e-6)
    result = dict(status="running", quick=args.quick, full_chord=args.full_chord,
                  sample_sha256=digest, sample_rate_hz=rate, sample_frames=len(signal),
                  signal_duration_s=duration, h_s=h, input_peak_v=.1,
                  controls=POSITIONS, speaker_parameters=resolved(), cases=[])
    def save():
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    save()
    try:
        old = circuit("resistive")
        with np.load(ROOT/"simulation/raw/linear_reduction/initial.npz") as stored:
            initial, start = stored["state"], float(stored["time"])
        axis = np.arange(len(signal))
        outputs = {}
        for kind in ("resistive", "reactive"):
            c = circuit(kind)
            seed = transfer_state(old, c, initial)
            warm, warm_stats = simulate(c, seed, start, .0002 if args.quick else .020, 20e-6)
            t0 = start + (.0002 if args.quick else .020)
            c.source_function("Vin", lambda t, origin=t0: .1*np.interp((t-origin)*rate, axis, signal, left=0., right=0.))
            states, stats = simulate(c, warm[-1], t0, duration, h)
            amp = diagnostics(c, states, h, MODEL["el34_grid_r"])
            row = dict(name=kind, unknowns=c.size, warm_stats=warm_stats, stats=stats,
                       amplifier=amp, load=load_metrics(c, states, h, kind))
            result["cases"].append(row); save()
            np.savez_compressed(RAW/f"{kind}.npz", state=states)
            sample_count = round(duration*rate)
            outputs[kind] = np.interp(np.arange(sample_count)/rate, np.arange(len(states))*h,
                                      states[:, c.index["out"]])
        scale = 1.05*max(np.max(np.abs(value)) for value in outputs.values())
        result["wav_common_scale_v"] = float(scale)
        for name, values in outputs.items():
            write_wav(DEST/f"{name}.wav", values, rate, scale)
        reference, candidate = outputs["resistive"], outputs["reactive"]
        gain = float(reference@candidate/(candidate@candidate))
        result["comparison"] = dict(
            output_relative_rms_difference=float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference)),
            output_shape_difference_after_gain=float(np.linalg.norm(gain*candidate-reference)/np.linalg.norm(reference)),
            reactive_gain_to_resistive=gain)
        frequencies = np.geomspace(20., 20_000., 601); z = impedance(frequencies)
        np.savetxt(DEST/"impedance.csv", np.c_[frequencies, z.real, z.imag, np.abs(z)], delimiter=",",
                   header="frequency_hz,real_ohm,imag_ohm,magnitude_ohm", comments="")
        result["status"] = "pass"; save(); render(result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}"); save(); render(result); raise


if __name__ == "__main__":
    main()
