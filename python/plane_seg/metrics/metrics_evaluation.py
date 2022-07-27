from typing import AnyStr, Dict, Tuple
from pathlib import Path
from evops.metrics import *
from . import rgb2labels

import cv2

__all__ = ["evaluate_metrics"]


def evaluate_metrics(
    prediction: NDArray[(Any, Any), np.int32],
    ground_truth_path: Path,
    metric_names: Tuple[AnyStr] = (
        "iou",
        "dice",
        "precision-iou",
        "recall-iou",
        "fScore-iou",
        "mean-iou",
        "mean-dice",
        "multivalue-0.8",  # Here user can specify any float between 0 and 1 as a threshold
    ),
    print_to_console: bool = True,
    output_file: Path = None,
) -> Dict[AnyStr, np.float64]:
    """
    Evaluates EVOPS metrics given predictions and ground truth

    :param prediction: an RGB image or a 2D NumPy array
    :param ground_truth_path: path to an RGB image or a 2D NumPy array, the same size as predictions
    :param metric_names: names of EVOPS metrics: any subset of: {iou, dice, precision-iou, recall-iou, fScore-iou, mean-iou, mean-dice, multivalue-0.X}. All by default
    :param print_to_console: prints to console if true
    :param output_file: appends output to the file if specified
    :return: the value of the metric
    """
    if not ground_truth_path.exists():
        raise ValueError("ground truth file not found")
    if ground_truth_path.suffix not in (".png", ".npy"):
        raise ValueError("ground truth may only be an RGB image or a NumPy array")

    if ground_truth_path.suffix == ".png":
        ground_truth = rgb2labels.rgb2labels(cv2.imread(str(ground_truth_path)))
    else:
        ground_truth = np.load(ground_truth_path)

    metric_values = {}

    for metric_name in metric_names:
        if metric_name == "iou":
            metric_values[metric_name] = iou(prediction, ground_truth)
        elif metric_name == "dice":
            metric_values[metric_name] = dice(prediction, ground_truth)
        elif metric_name == "precision-iou":
            metric_values[metric_name] = precision(prediction, ground_truth, "iou")
        elif metric_name == "recall-iou":
            metric_values[metric_name] = recall(prediction, ground_truth, "iou")
        elif metric_name == "fScore-iou":
            metric_values[metric_name] = fScore(prediction, ground_truth, "iou")
        elif metric_name == "mean-iou":
            metric_values[metric_name] = mean(prediction, ground_truth, iou)
        elif metric_name == "mean-dice":
            metric_values[metric_name] = mean(prediction, ground_truth, dice)
        elif metric_name.startswith("multivalue"):
            splitted = metric_name.split("-")
            if len(splitted) == 2:
                threshold = np.float64(splitted[1])
                if not (0 <= threshold <= 1):
                    raise ValueError(f"Invalid multivalue threshold")
            else:
                threshold = np.float64(0.8)
            multivalue_result = multi_value(prediction, ground_truth, threshold)

            metric_values["under_segmented"] = multivalue_result["under_segmented"]
            metric_values["over_segmented"] = multivalue_result["over_segmented"]
            metric_values["missed"] = multivalue_result["missed"]
            metric_values["noise"] = multivalue_result["noise"]
        else:
            raise ValueError(f"Invalid metric name {metric_name}.")

    if print_to_console:
        for metric, value in metric_values.items():
            print(f"{metric:15} {value}")
    if output_file is not None:
        with open(output_file, "w") as fout:
            for metric, value in metric_values.items():
                fout.write(f"{metric:15} {value}\n")
            fout.write("\n")

    return metric_values
