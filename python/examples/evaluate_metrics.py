from argparse import ArgumentParser
from pathlib import Path
from python.plane_seg.metrics import evaluate_metrics

import sys
import numpy as np


def main(argv):
    parser = ArgumentParser()
    parser.add_argument(
        "--print-to-console",
        help="optional, true by default, can be one of: true, false",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="optional, path to an output file.\nif specified, metric value is appended there",
    )

    args = parser.parse_args(argv)

    predictions = np.load("data/metrics_example_predictions.npy")

    evaluate_metrics(
        predictions,
        Path("data/metrics_example_ground_truth.png"),
        metric_names=("precision-iou", "mean-dice", "multivalue-0.8"),
        print_to_console=(args.print_to_console != "false"),
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
