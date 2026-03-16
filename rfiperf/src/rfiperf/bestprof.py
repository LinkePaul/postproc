from pathlib import Path
import re

import numpy as np


def _parse_float(text):
    text = text.strip()
    if text in {"N/A", "nan", "NaN"}:
        return None

    first = text.split("+/-", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def _parse_prob_noise(text):
    text = text.strip()
    if not text:
        return None

    first = text.split("(", 1)[0].strip()
    return first or None


def _parse_sigma(text):
    m = re.search(r"~\s*([0-9.+\-eE]+)\s*sigma", text)
    if not m:
        return None
    return float(m.group(1))


def parse_bestprof(path):
    path = Path(path)

    header = {}
    profile = []

    for line in path.read_text().splitlines():
        if line.startswith("#"):
            # Standard PRESTO header line: "# Key = value"
            m = re.match(r"#\s*(.*?)\s*=\s*(.*)$", line)
            if m:
                key = m.group(1).strip()
                value = m.group(2).strip()
                header[key] = value
                continue

            # Special PRESTO line without "=":
            # "# Prob(Noise)      <  0   (~53.4 sigma)"
            m = re.match(r"#\s*Prob\(Noise\)\s+(.*)$", line)
            if m:
                header["Prob(Noise)"] = m.group(1).strip()
                continue

            continue

        if not line.strip():
            continue

        if set(line.strip()) == {"#"}:
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        profile.append(float(parts[1]))

    profile = np.asarray(profile, dtype=float)

    if profile.size == 0:
        raise ValueError(f"No profile bins found in {path}")

    prob_noise_text = header.get("Prob(Noise)", "")

    out = {
        "path": str(path),
        "input_file": header.get("Input file"),
        "candidate": header.get("Candidate"),
        "profile_bins": int(header.get("Profile Bins", profile.size)),
        "profile_avg": _parse_float(header.get("Profile Avg", "")),
        "profile_stddev": _parse_float(header.get("Profile StdDev", "")),
        "reduced_chi_sqr": _parse_float(header.get("Reduced chi-sqr", "")),
        "best_dm": _parse_float(header.get("Best DM", "")),
        "p_topo_ms": _parse_float(header.get("P_topo (ms)", "")),
        "p_bary_ms": _parse_float(header.get("P_bary (ms)", "")),
        "prob_noise": _parse_prob_noise(prob_noise_text),
        "presto_sigma": _parse_sigma(prob_noise_text),
        "profile": profile,
        "header": header,
    }
    return out


def profile_snr(profile, baseline="median"):
    profile = np.asarray(profile, dtype=float)

    if baseline == "median":
        base = float(np.median(profile))
    elif baseline == "mean":
        base = float(np.mean(profile))
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    noise = float(np.std(profile))
    peak = float(np.max(profile))

    if noise == 0.0:
        snr = 0.0
    else:
        snr = (peak - base) / noise

    return {
        "baseline": baseline,
        "baseline_value": base,
        "noise_std": noise,
        "peak_value": peak,
        "peak_bin": int(np.argmax(profile)),
        "profile_snr": snr,
    }


def summarize_bestprof(path, baseline="median"):
    data = parse_bestprof(path)
    metrics = profile_snr(data["profile"], baseline=baseline)

    return {
        "path": data["path"],
        "input_file": data["input_file"],
        "candidate": data["candidate"],
        "profile_bins": data["profile_bins"],
        "best_dm": data["best_dm"],
        "p_topo_ms": data["p_topo_ms"],
        "p_bary_ms": data["p_bary_ms"],
        "reduced_chi_sqr": data["reduced_chi_sqr"],
        "prob_noise": data["prob_noise"],
        "presto_sigma": data["presto_sigma"],
        **metrics,
    }


def load_summaries(paths, baseline="median"):
    return [summarize_bestprof(path, baseline=baseline) for path in paths]


def summarize_comparison(items, baseline="median"):
    if not items:
        raise ValueError("No items to compare")

    items = sorted(items, key=lambda x: x["profile_snr"], reverse=True)

    profile_snrs = [x["profile_snr"] for x in items]
    chi_vals = [x["reduced_chi_sqr"] for x in items if x["reduced_chi_sqr"] is not None]
    sigma_vals = [x["presto_sigma"] for x in items if x["presto_sigma"] is not None]

    best_by_profile_snr = max(items, key=lambda x: x["profile_snr"])
    best_by_reduced_chi = max(
        items,
        key=lambda x: x["reduced_chi_sqr"] if x["reduced_chi_sqr"] is not None else float("-inf"),
    )

    return {
        "mode": "compare",
        "baseline": baseline,
        "n_files": len(items),
        "best_by_profile_snr": best_by_profile_snr["path"],
        "best_by_reduced_chi_sqr": best_by_reduced_chi["path"],
        "profile_snr": {
            "min": float(np.min(profile_snrs)),
            "max": float(np.max(profile_snrs)),
            "mean": float(np.mean(profile_snrs)),
            "median": float(np.median(profile_snrs)),
        },
        "reduced_chi_sqr": {
            "min": float(np.min(chi_vals)) if chi_vals else None,
            "max": float(np.max(chi_vals)) if chi_vals else None,
            "mean": float(np.mean(chi_vals)) if chi_vals else None,
            "median": float(np.median(chi_vals)) if chi_vals else None,
        },
        "presto_sigma": {
            "min": float(np.min(sigma_vals)) if sigma_vals else None,
            "max": float(np.max(sigma_vals)) if sigma_vals else None,
            "mean": float(np.mean(sigma_vals)) if sigma_vals else None,
            "median": float(np.median(sigma_vals)) if sigma_vals else None,
        },
        "files": items,
    }


def format_comparison_table(items):
    items = sorted(items, key=lambda x: x["profile_snr"], reverse=True)

    rows = []
    header = (
        f"{'file':40} {'dm':>8} {'p_bary_ms':>14} "
        f"{'chi_red':>10} {'p_sigma':>10} {'profile_snr':>12}"
    )
    rows.append(header)
    rows.append("-" * len(header))

    for item in items:
        file_label = Path(item["path"]).parent.name
        dm = f"{item['best_dm']:.3f}" if item["best_dm"] is not None else "None"
        p_bary = f"{item['p_bary_ms']:.6f}" if item["p_bary_ms"] is not None else "None"
        chi = f"{item['reduced_chi_sqr']:.3f}" if item["reduced_chi_sqr"] is not None else "None"
        psig = f"{item['presto_sigma']:.1f}" if item["presto_sigma"] is not None else "None"
        psnr = f"{item['profile_snr']:.3f}"

        rows.append(
            f"{file_label:40} {dm:>8} {p_bary:>14} "
            f"{chi:>10} {psig:>10} {psnr:>12}"
        )

    return "\n".join(rows)
