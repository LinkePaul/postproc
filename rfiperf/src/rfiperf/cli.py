import argparse
import json
import os
from pathlib import Path

from postproc_common.configio import label_for_path, load_config_for_paths
from postproc_common.kurtio import read_mask

from .bestprof import (
    format_comparison_table,
    load_summaries,
    parse_bestprof,
    summarize_comparison,
)
from .kurtosis import (
    ant_label_for_index,
    build_waterfall_axis_info,
    load_spliced_mask_from_file,
    load_spliced_mask_from_obs_dir,
    resolve_ant_index,
    resolve_kurtosis_input,
    select_pol,
    stream_extract_waterfall,
    stream_extract_waterfalls,
    summary_stats,
    zap_fraction_over_ant,
    zap_fraction_over_freq,
    zap_fraction_over_time,
)
from .plotting import (
    save_profile_overlay,
    save_profile_plot,
    save_waterfall,
    save_waterfall_grid,
    save_zap_fraction_over_ant,
    save_zap_fraction_over_freq,
    save_zap_fraction_over_time,
)


def make_plot_path(outdir, lo, pol, kind, ant=None):
    name = f"{lo or 'mask'}_pol{pol}"
    if ant is not None:
        name += f"_ant{ant}"
    name += f"_{kind}.png"
    return outdir / name


def make_waterfall_plot_path(
    outdir,
    lo,
    pol,
    ant=None,
    tstart=None,
    tend=None,
    fstart=None,
    fend=None,
):
    name = f"{lo or 'mask'}_pol{pol}"
    if ant is not None:
        name += f"_ant{ant}"

    if tstart is not None or tend is not None:
        name += f"_t{0 if tstart is None else tstart}-{'end' if tend is None else tend}"

    if fstart is not None or fend is not None:
        name += f"_f{0 if fstart is None else fstart}-{'end' if fend is None else fend}"

    name += "_waterfall.png"
    return outdir / name


def make_snr_plot_path(outdir, stem, kind):
    return outdir / f"{stem}_{kind}.png"


def main():
    parser = argparse.ArgumentParser(prog="rfiperf")
    sub = parser.add_subparsers(dest="command", required=True)

    p_kurt = sub.add_parser("kurtosis")
    p_kurt.add_argument("input_path", metavar="INPUT")
    p_kurt.add_argument("--lo")
    p_kurt.add_argument("--status")
    p_kurt.add_argument("--pol", required=True, choices=["x", "y", "xy"])
    p_kurt.add_argument("--json", choices=["summary"])
    p_kurt.add_argument("--plot", choices=["freq", "ant", "time", "waterfall", "all"])
    p_kurt.add_argument("--ant")
    p_kurt.add_argument("--tstart", type=int)
    p_kurt.add_argument("--tend", type=int)
    p_kurt.add_argument("--fstart", type=int)
    p_kurt.add_argument("--fend", type=int)
    p_kurt.add_argument("--kbsize", type=int, default=256)
    p_kurt.add_argument("--outdir")

    p_snr = sub.add_parser("snr")
    p_snr.add_argument("inputs", nargs="+", metavar="INPUT")
    p_snr.add_argument("--baseline", choices=["mean", "median"], default="median")
    p_snr.add_argument("--json", action="store_true")
    p_snr.add_argument("--plot", choices=["profile", "overlay"])
    p_snr.add_argument("--normalize", action="store_true")
    p_snr.add_argument("--outdir")

    args = parser.parse_args()

    if args.command == "kurtosis":
        input_path = Path(args.input_path).expanduser()
        config = load_config_for_paths(paths=[input_path], cwd=Path.cwd())
        file_label = label_for_path(input_path, config)

        outdir = (
            Path(args.outdir).expanduser()
            if args.outdir
            else (
                input_path / "rfiperf_out"
                if input_path.is_dir()
                else input_path.parent / "rfiperf_out"
            )
        )
        outdir.mkdir(parents=True, exist_ok=True)

        mask_path, lo, layout = resolve_kurtosis_input(
            input_path,
            lo=args.lo,
            status_path=args.status,
            kbsize=args.kbsize,
        )

        if lo is None:
            lo = "mask"

        ant_idx = None
        ant_label = None
        if args.ant is not None:
            try:
                ant_idx = resolve_ant_index(layout, args.ant)
            except ValueError as e:
                raise SystemExit(str(e))
            ant_label = ant_label_for_index(layout, ant_idx)

        if args.plot == "waterfall":
            axis_info = build_waterfall_axis_info(
                layout,
                tstart=args.tstart,
                fstart=args.fstart,
            )

            if ant_idx is not None:
                data, _ = stream_extract_waterfall(
                    mask_path,
                    layout,
                    args.pol,
                    ant=ant_idx,
                    tstart=args.tstart,
                    tend=args.tend,
                    fstart=args.fstart,
                    fend=args.fend,
                )

                out_path = make_waterfall_plot_path(
                    outdir,
                    lo,
                    args.pol,
                    ant=ant_label,
                    tstart=args.tstart,
                    tend=args.tend,
                    fstart=args.fstart,
                    fend=args.fend,
                )

                save_waterfall(
                    data,
                    out_path,
                    lo,
                    args.pol,
                    ant=ant_label,
                    tstart=args.tstart,
                    tend=args.tend,
                    fstart=args.fstart,
                    fend=args.fend,
                    axis_info=axis_info,
                )
                print(out_path)
                return

            data_list, ant_idx_list = stream_extract_waterfalls(
                mask_path,
                layout,
                args.pol,
                ants=None,
                tstart=args.tstart,
                tend=args.tend,
                fstart=args.fstart,
                fend=args.fend,
            )
            ant_labels = [ant_label_for_index(layout, i) for i in ant_idx_list]

            out_path = make_waterfall_plot_path(
                outdir,
                lo,
                args.pol,
                ant=None,
                tstart=args.tstart,
                tend=args.tend,
                fstart=args.fstart,
                fend=args.fend,
            )

            save_waterfall_grid(
                data_list,
                ant_labels,
                out_path,
                lo,
                args.pol,
                tstart=args.tstart,
                tend=args.tend,
                fstart=args.fstart,
                fend=args.fend,
                axis_info=axis_info,
            )
            print(out_path)
            return

        mask = read_mask(mask_path, layout)
        mask_pol = select_pol(mask, args.pol)

        if args.json == "summary":
            out = {
                "lo": lo,
                "pol": args.pol,
                "file_label": file_label,
                **summary_stats(mask_pol, ant=ant_idx),
            }
            if ant_label is not None:
                out["ant"] = ant_label
            print(json.dumps(out, indent=2))
            return

        if args.plot == "freq":
            values = zap_fraction_over_freq(mask_pol, ant=ant_idx)
            out_path = make_plot_path(outdir, lo, args.pol, "freq", ant=ant_label)
            save_zap_fraction_over_freq(values, out_path, lo, args.pol, ant=ant_label)
            print(out_path)
            return

        if args.plot == "ant":
            values = zap_fraction_over_ant(mask_pol)
            out_path = make_plot_path(outdir, lo, args.pol, "ant")
            save_zap_fraction_over_ant(values, out_path, lo, args.pol)
            print(out_path)
            return

        if args.plot == "time":
            values = zap_fraction_over_time(mask_pol, ant=ant_idx)
            out_path = make_plot_path(outdir, lo, args.pol, "time", ant=ant_label)
            save_zap_fraction_over_time(values, out_path, lo, args.pol, ant=ant_label)
            print(out_path)
            return

        if args.plot == "all":
            freq_values = zap_fraction_over_freq(mask_pol, ant=ant_idx)
            ant_values = zap_fraction_over_ant(mask_pol)
            time_values = zap_fraction_over_time(mask_pol, ant=ant_idx)

            freq_path = make_plot_path(outdir, lo, args.pol, "freq", ant=ant_label)
            ant_path = make_plot_path(outdir, lo, args.pol, "ant")
            time_path = make_plot_path(outdir, lo, args.pol, "time", ant=ant_label)

            save_zap_fraction_over_freq(freq_values, freq_path, lo, args.pol, ant=ant_label)
            save_zap_fraction_over_ant(ant_values, ant_path, lo, args.pol)
            save_zap_fraction_over_time(time_values, time_path, lo, args.pol, ant=ant_label)

            print(freq_path)
            print(ant_path)
            print(time_path)
            return

        raise SystemExit("Use either --json summary or --plot ...")

    if args.command == "snr":
        input_paths = [Path(p).expanduser() for p in args.inputs]
        missing = [str(p) for p in input_paths if not p.exists()]
        if missing:
            raise SystemExit("Input file(s) not found: " + ", ".join(missing))

        compare_mode = len(input_paths) > 1
        config = load_config_for_paths(paths=input_paths, cwd=Path.cwd())

        outdir = None
        if args.outdir:
            outdir = Path(args.outdir).expanduser()
            outdir.mkdir(parents=True, exist_ok=True)
        elif compare_mode:
            common_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))
            outdir = common_root / "rfiperf_compare"
            outdir.mkdir(parents=True, exist_ok=True)

        summaries = load_summaries(input_paths, baseline=args.baseline)
        for item in summaries:
            item["file_label"] = label_for_path(item["path"], config)

        if compare_mode:
            summaries.sort(key=lambda x: x["profile_snr"], reverse=True)

            if args.json:
                print(json.dumps(summarize_comparison(summaries, baseline=args.baseline), indent=2))
            elif args.plot is None:
                print(format_comparison_table(summaries))

            if args.plot == "overlay":
                profiles = [parse_bestprof(p)["profile"] for p in input_paths]
                labels = [label_for_path(p, config) for p in input_paths]

                if outdir is None:
                    common_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))
                    outdir = common_root / "rfiperf_compare"
                    outdir.mkdir(parents=True, exist_ok=True)

                out_path = make_snr_plot_path(outdir, "compare", "overlay")
                save_profile_overlay(profiles, labels, out_path, normalize=args.normalize)
                print(out_path)
            elif args.plot == "profile":
                raise SystemExit("--plot profile is only valid for a single input")
            return

        summary = summaries[0]

        if args.json or args.plot is None:
            print(json.dumps(summary, indent=2))

        if args.plot == "profile":
            data = parse_bestprof(input_paths[0])

            if outdir is None:
                outdir = input_paths[0].parent / "rfiperf_out"
                outdir.mkdir(parents=True, exist_ok=True)

            out_path = make_snr_plot_path(outdir, input_paths[0].stem, "profile")
            save_profile_plot(data["profile"], out_path, summary["file_label"])
            print(out_path)
        elif args.plot == "overlay":
            raise SystemExit("--plot overlay requires multiple inputs")
        return
