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


def profile_snr(profile, baseline="median", threshold_sigma=1.5, min_width_bins=1):
    profile = np.asarray(profile, dtype=float)
    nbin = len(profile)

    peak_bin = int(np.argmax(profile))

    crude_half_width = max(1, nbin // 32)
    crude_idx = np.array(
        [(peak_bin + offset) % nbin for offset in range(-crude_half_width, crude_half_width + 1)],
        dtype=int,
    )

    offpulse_mask = np.ones(nbin, dtype=bool)
    offpulse_mask[crude_idx] = False
    offpulse = profile[offpulse_mask]

    if offpulse.size == 0:
        raise ValueError("No off-pulse bins left for first-pass noise estimate")

    if baseline == "median":
        base = float(np.median(offpulse))
    elif baseline == "mean":
        base = float(np.mean(offpulse))
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    noise = float(np.std(offpulse))
    if noise == 0.0:
        return {
            "baseline": baseline,
            "baseline_value": base,
            "noise_std": noise,
            "peak_value": float(np.max(profile)),
            "peak_bin": peak_bin,
            "pulse_bins": [peak_bin],
            "pulse_nbin": 1,
            "integrated_signal": 0.0,
            "profile_snr": 0.0,
        }

    threshold = base + threshold_sigma * noise

    pulse_bins = {peak_bin}

    i = peak_bin
    while True:
        j = (i - 1) % nbin
        if j == peak_bin or profile[j] <= threshold:
            break
        pulse_bins.add(j)
        i = j

    i = peak_bin
    while True:
        j = (i + 1) % nbin
        if j == peak_bin or profile[j] <= threshold:
            break
        pulse_bins.add(j)
        i = j

    pulse_bins = sorted(pulse_bins)

    if len(pulse_bins) < min_width_bins:
        half = max(0, min_width_bins // 2)
        pulse_bins = sorted({(peak_bin + offset) % nbin for offset in range(-half, half + 1)})

    pulse_idx = np.array(pulse_bins, dtype=int)

    offpulse_mask = np.ones(nbin, dtype=bool)
    offpulse_mask[pulse_idx] = False
    offpulse = profile[offpulse_mask]

    if offpulse.size == 0:
        raise ValueError("No off-pulse bins left after adaptive pulse selection")

    if baseline == "median":
        base = float(np.median(offpulse))
    elif baseline == "mean":
        base = float(np.mean(offpulse))

    noise = float(np.std(offpulse))

    signal = float(np.sum(profile[pulse_idx] - base))
    snr = 0.0 if noise == 0.0 else signal / (noise * np.sqrt(len(pulse_idx)))

    return {
        "baseline": baseline,
        "baseline_value": base,
        "noise_std": noise,
        "threshold_sigma": threshold_sigma,
        "threshold_value": threshold,
        "peak_value": float(np.max(profile)),
        "peak_bin": peak_bin,
        "pulse_bins": pulse_idx.tolist(),
        "pulse_nbin": int(len(pulse_idx)),
        "integrated_signal": signal,
        "profile_snr": float(snr),
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

    headers = ["file", "dm", "p_bary_ms", "chi_red", "p_sigma", "prof_snr"]
    rows = []

    for item in items:
        file_label = item.get("file_label", Path(item["path"]).parent.name)
        dm = f"{item['best_dm']:.3f}" if item["best_dm"] is not None else "None"
        p_bary = f"{item['p_bary_ms']:.6f}" if item["p_bary_ms"] is not None else "None"
        chi = f"{item['reduced_chi_sqr']:.3f}" if item["reduced_chi_sqr"] is not None else "None"
        psig = f"{item['presto_sigma']:.1f}" if item["presto_sigma"] is not None else "None"
        psnr = f"{item['profile_snr']:.3f}"

        rows.append([file_label, dm, p_bary, chi, psig, psnr])

    widths = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            max_len = max(max_len, len(row[col_idx]))
        widths.append(max_len + 3)

    lines = []

    header_line = (
        f"{headers[0]:<{widths[0]}}"
        f"{headers[1]:>{widths[1]}}"
        f"{headers[2]:>{widths[2]}}"
        f"{headers[3]:>{widths[3]}}"
        f"{headers[4]:>{widths[4]}}"
        f"{headers[5]:>{widths[5]}}"
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        line = (
            f"{row[0]:<{widths[0]}}"
            f"{row[1]:>{widths[1]}}"
            f"{row[2]:>{widths[2]}}"
            f"{row[3]:>{widths[3]}}"
            f"{row[4]:>{widths[4]}}"
            f"{row[5]:>{widths[5]}}"
        )
        lines.append(line)

    return "\n".join(lines)

def center_profile_on_peak(profile):
    profile = np.asarray(profile, dtype=float)
    nbin = len(profile)
    peak_bin = int(np.argmax(profile))
    center_bin = nbin // 2
    shift = center_bin - peak_bin
    centered = np.roll(profile, shift)
    return {
        "profile": centered,
        "peak_bin": peak_bin,
        "center_bin": center_bin,
        "shift": shift,
    }