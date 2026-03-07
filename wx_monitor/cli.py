"""
wx_monitor.cli
==============
Command-line interface for running wx_monitor products.

Usage examples::

    # Run all three products
    python -m wx_monitor

    # Run a single product
    python -m wx_monitor goes_mrms
    python -m wx_monitor mrms_hrrr
    python -m wx_monitor hrrr_asos

    # Print the shared valid time and exit
    python -m wx_monitor --valid-time

Options::

    --output-dir DIR   Directory for PNG/CSV output (default: ./output)
    --n-hours N        Number of back-hours for hrrr_asos frames (default: 12)
    --valid-time       Print the shared valid time and exit
"""

from __future__ import annotations

import argparse
import sys


from wx_monitor.config import VALID_TIME


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wx_monitor",
        description="Operational weather visualisation — wx_monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "product",
        nargs="?",
        choices=["goes_mrms", "mrms_hrrr", "hrrr_asos", "refl_m10"],
        default=None,
        help="Product to run (omit to run all three)",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for all output files (default: output/)",
    )
    p.add_argument(
        "--n-hours",
        type=int,
        default=12,
        metavar="N",
        help="Back-hours for hrrr_asos frames (default: 12)",
    )
    p.add_argument(
        "--valid-time",
        action="store_true",
        help="Print the shared valid time and exit",
    )
    p.add_argument(
        "--manifest",
        action="store_true",
        help="Write manifest.json without running any products, then exit",
    )
    p.add_argument(
        "--manifest-path",
        default="manifest.json",
        metavar="PATH",
        help="Path to write manifest.json (default: manifest.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print(
        f"\n{'━'*70}\n"
        f"  wx_monitor  —  valid time anchor: "
        f"{VALID_TIME.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'━'*70}\n"
    )

    if args.valid_time:
        print(f"VALID_TIME = {VALID_TIME.isoformat()}")
        return 0

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # ── manifest-only mode ─────────────────────────────────────────────────
    if args.manifest:
        from wx_monitor.manifest import write_manifest
        write_manifest(output_dir=args.output_dir,
                       manifest_path=args.manifest_path)
        return 0

    products = (
        [args.product] if args.product
        else ["goes_mrms", "mrms_hrrr", "hrrr_asos", "refl_m10"]
    )

    errors: list[str] = []

    for name in products:
        print(f"\n{'▶'*3}  Running: {name}  {'◀'*3}")
        try:
            if name == "goes_mrms":
                from wx_monitor import goes_mrms
                goes_mrms.run(
                    output_path=os.path.join(args.output_dir,
                                             "goes19_conus_geocolor_mrms.png")
                )

            elif name == "mrms_hrrr":
                from wx_monitor import mrms_hrrr
                mrms_hrrr.run(output_dir=args.output_dir)

            elif name == "hrrr_asos":
                from wx_monitor import hrrr_asos
                frames_dir = os.path.join(args.output_dir, "frames")
                hrrr_asos.run(n_hours=args.n_hours, output_dir=frames_dir)

            elif name == "refl_m10":
                from wx_monitor import refl_m10
                refl_m10.run(output_dir=args.output_dir)

        except Exception as exc:
            import traceback
            msg = f"{name} failed: {exc}"
            errors.append(msg)
            print(f"\n  ✗ {msg}")
            traceback.print_exc()

    if errors:
        print(f"\n{'─'*70}")
        print(f"  {len(errors)} product(s) failed:")
        for e in errors:
            print(f"    • {e}")
        return 1

    # ── write manifest so the HTML viewer can auto-discover outputs ────────
    print(f"\n{'▶'*3}  Writing manifest  {'◀'*3}")
    try:
        from wx_monitor.manifest import write_manifest
        write_manifest(output_dir=args.output_dir,
                       manifest_path=args.manifest_path)
    except Exception as exc:
        print(f"  ⚠ manifest not written: {exc}")

    print(f"\n{'─'*70}")
    print(f"  All products completed.  Output → {args.output_dir}/")
    print(f"{'─'*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
