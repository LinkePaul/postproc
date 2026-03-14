import argparse
from pathlib import Path

from postproc_common.discovery import discover_slices
from postproc_common.metadata import load_all_metadata
from postproc_common.kurtio import stream_concat_masks


def main():
    parser = argparse.ArgumentParser(
        prog="kurtsplice",
        description="Concatenate ATA kurtosismask.bin files for one LO",
    )
    parser.add_argument(
        "obs_root",
        metavar="INPUT_DIR",
        help="Path to observation directory containing LoX.C#### subdirectories",
    )
    parser.add_argument(
        "--lo",
        required=True,
        metavar="LO",
        help="Local oscillator to merge, e.g. A or LoA",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_FILE",
        help="Output file path. Default: INPUT_DIR/LoX_spliced.kurtosismask.bin",
    )
    parser.add_argument(
        "--kbsize",
        type=int,
        default=256,
        metavar="N",
        help="Kurtosis channel/block parameter used by the writer, usually KURTCHNL (default: 256)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if input mask files do not all have the same block count",
    )

    args = parser.parse_args()

    obs_root = Path(args.obs_root).expanduser()

    slices = discover_slices(obs_root, lo=args.lo)
    if not slices:
        raise SystemExit(f"No slice directories found for LO {args.lo}")

    missing = [s for s in slices if s["kurt_path"] is None]
    if missing:
        msg = "\n".join(str(s["path"]) for s in missing)
        raise SystemExit(f"Missing kurtosismask.bin in:\n{msg}")

    meta_list = load_all_metadata(slices)

    lo_name = slices[0]["lo"]

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = obs_root / f"{lo_name}_spliced.kurtosismask.bin"

    stream_concat_masks(
        slices,
        meta_list,
        out_path,
        kbsize=args.kbsize,
        strict=args.strict,
    )

    print(out_path)


if __name__ == "__main__":
    main()