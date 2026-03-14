import argparse
import json
from pathlib import Path

from .kurtosis import (
    load_spliced_mask_from_obs_dir,
    load_spliced_mask_from_file,
    select_pol,
    summary_stats,
    zap_fraction_over_freq,
    zap_fraction_over_ant,
    zap_fraction_over_time,
)
from .plotting import (
    save_zap_fraction_over_freq,
    save_zap_fraction_over_ant,
    save_zap_fraction_over_time,
)


def make_plot_path(outdir, lo, pol, kind, ant=None):
    name = f"{lo or 'mask'}_pol{pol}"
    if ant is not None:
        name += f"_ant{ant}"
    name += f"_{kind}.png"
    return outdir / name


def main():
    parser = argparse.ArgumentParser(prog="rfiperf")
    sub = parser.add_subparsers(dest="command", required=True)

    p_kurt = sub.add_parser("kurtosis")
    p_kurt.add_argument("input_path", metavar="INPUT")
    p_kurt.add_argument("--lo")
    p_kurt.add_argument("--status")
    p_kurt.add_argument("--nchan", type=int)
    p_kurt.add_argument("--pol", required=True, choices=["x", "y", "xy"])
    p_kurt.add_argument("--json", choices=["summary"])
    p_kurt.add_argument("--plot", choices=["freq", "ant", "time", "all"])
    p_kurt.add_argument("--ant", type=int)
    p_kurt.add_argument("--kbsize", type=int, default=256)
    p_kurt.add_argument("--outdir")

    args = parser.parse_args()

    if args.command != "kurtosis":
        return

    input_path = Path(args.input_path).expanduser()

    if input_path.is_dir():
        if not args.lo:
            raise SystemExit("--lo is required when INPUT is an observation directory")

        obs_root = input_path
        outdir = Path(args.outdir).expanduser() if args.outdir else obs_root / "rfiperf_out"
        outdir.mkdir(parents=True, exist_ok=True)

        mask, lo, layout = load_spliced_mask_from_obs_dir(obs_root, args.lo, kbsize=args.kbsize)

    else:
        if args.status is None:
            raise SystemExit("--status is required when INPUT is a .kurtosismask.bin file")
        if args.nchan is None:
            raise SystemExit("--nchan is required when INPUT is a .kurtosismask.bin file")

        mask_path = input_path
        outdir = Path(args.outdir).expanduser() if args.outdir else mask_path.parent / "rfiperf_out"
        outdir.mkdir(parents=True, exist_ok=True)

        mask, lo, layout = load_spliced_mask_from_file(
            mask_path,
            args.status,
            args.nchan,
            kbsize=args.kbsize,
        )

        if lo is None:
            lo = "mask"

    mask_pol = select_pol(mask, args.pol)

    if args.ant is not None:
        if args.ant < 0 or args.ant >= mask_pol.shape[0]:
            raise SystemExit(
                f"Invalid antenna index {args.ant}, valid range is 0 to {mask_pol.shape[0] - 1}"
            )

    if args.json == "summary":
        out = {
            "lo": lo,
            "pol": args.pol,
            **summary_stats(mask_pol, ant=args.ant),
        }
        print(json.dumps(out, indent=2))
        return

    if args.plot == "freq":
        values = zap_fraction_over_freq(mask_pol, ant=args.ant)
        out_path = make_plot_path(outdir, lo, args.pol, "freq", ant=args.ant)
        save_zap_fraction_over_freq(values, out_path, lo, args.pol, ant=args.ant)
        print(out_path)
        return

    if args.plot == "ant":
        values = zap_fraction_over_ant(mask_pol)
        out_path = make_plot_path(outdir, lo, args.pol, "ant")
        save_zap_fraction_over_ant(values, out_path, lo, args.pol)
        print(out_path)
        return

    if args.plot == "time":
        values = zap_fraction_over_time(mask_pol, ant=args.ant)
        out_path = make_plot_path(outdir, lo, args.pol, "time", ant=args.ant)
        save_zap_fraction_over_time(values, out_path, lo, args.pol, ant=args.ant)
        print(out_path)
        return

    if args.plot == "all":
        freq_values = zap_fraction_over_freq(mask_pol, ant=args.ant)
        ant_values = zap_fraction_over_ant(mask_pol)
        time_values = zap_fraction_over_time(mask_pol, ant=args.ant)

        freq_path = make_plot_path(outdir, lo, args.pol, "freq", ant=args.ant)
        ant_path = make_plot_path(outdir, lo, args.pol, "ant")
        time_path = make_plot_path(outdir, lo, args.pol, "time", ant=args.ant)

        save_zap_fraction_over_freq(freq_values, freq_path, lo, args.pol, ant=args.ant)
        save_zap_fraction_over_ant(ant_values, ant_path, lo, args.pol)
        save_zap_fraction_over_time(time_values, time_path, lo, args.pol, ant=args.ant)

        print(freq_path)
        print(ant_path)
        print(time_path)
        return

    raise SystemExit("Use either --json summary or --plot ...")