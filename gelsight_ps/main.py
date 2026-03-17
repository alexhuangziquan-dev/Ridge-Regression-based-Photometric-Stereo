"""Entry point for the gelsight_ps package.

Dispatches to the calibration or reconstruction pipeline based on the
``--mode`` command-line argument.

Typical usage::

    python -m gelsight_ps.main --config configs/example_config.py --mode calib
    python -m gelsight_ps.main --config configs/example_config.py --mode solve
"""

import argparse
from .config import load_config
from .calib.ball_calibrate import run_ball_calibration


def main():
    """Parses command-line arguments and runs the selected pipeline mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["calib", "solve"], default="calib")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "calib":
        run_ball_calibration(cfg)
    else:
        # Deferred import: avoids pulling in solve dependencies during calib mode.
        from .solve.reconstruct import run_reconstruction
        run_reconstruction(cfg)


if __name__ == "__main__":
    main()
