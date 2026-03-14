from pathlib import Path


def normalize_lo(lo):
    if lo is None:
        return None
    lo = lo.strip()
    if not lo:
        return None
    if lo.startswith("Lo"):
        return lo
    return "Lo" + lo.upper()


def parse_slice_dir_name(name):
    parts = name.split(".")
    if len(parts) != 2:
        return None

    lo, subband = parts

    if len(lo) != 3 or not lo.startswith("Lo"):
        return None
    if not subband.startswith("C"):
        return None

    try:
        return lo, int(subband[1:])
    except ValueError:
        return None


def pick_optional(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches in {directory} for {pattern}: {matches}")
    if matches:
        return matches[0]
    return None


def slice_sort_key(item):
    return item["lo"], item["schan"]


def discover_slices(obs_root, lo=None):
    obs_root = Path(obs_root)
    want_lo = normalize_lo(lo)

    if not obs_root.is_dir():
        raise NotADirectoryError(obs_root)

    out = []

    for entry in sorted(obs_root.iterdir()):
        if not entry.is_dir():
            continue

        parsed = parse_slice_dir_name(entry.name)
        if parsed is None:
            continue

        entry_lo, schan = parsed
        if want_lo is not None and entry_lo != want_lo:
            continue

        status_path = entry / "status_dump.json"
        if not status_path.exists():
            status_path = None

        out.append({
            "path": entry,
            "lo": entry_lo,
            "schan": schan,
            "fil_path": pick_optional(entry, "*.fil"),
            "kurt_path": pick_optional(entry, "*.kurtosismask.bin"),
            "status_path": status_path,
        })

    out.sort(key=slice_sort_key)
    return out