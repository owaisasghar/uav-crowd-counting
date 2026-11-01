from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_point_sets(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return distances for a maximum-cardinality one-to-one radius match."""
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 2)
    ground_truth = np.asarray(ground_truth, dtype=np.float32).reshape(-1, 2)
    if len(predicted) == 0 or len(ground_truth) == 0:
        return np.empty((0,), dtype=np.float32)
    distances = np.linalg.norm(
        predicted[:, None, :] - ground_truth[None, :, :],
        axis=2,
    )
    # An invalid assignment must cost more than every possible collection of
    # valid assignments so Hungarian matching maximizes valid cardinality first.
    invalid_cost = (max(len(predicted), len(ground_truth)) + 1) * (radius + 1.0)
    costs = np.where(distances <= radius, distances, invalid_cost)
    predicted_indices, target_indices = linear_sum_assignment(costs)
    selected = distances[predicted_indices, target_indices]
    return selected[selected <= radius].astype(np.float32)


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float | int]:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def localization_summary(
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    radius: float,
) -> dict[str, float | int | None]:
    if len(predictions) != len(ground_truth):
        raise ValueError("Prediction and ground-truth image counts differ")
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_distances: list[float] = []
    for predicted, target in zip(predictions, ground_truth):
        matches = match_point_sets(predicted, target, radius)
        true_positives += len(matches)
        false_positives += len(predicted) - len(matches)
        false_negatives += len(target) - len(matches)
        matched_distances.extend(matches.tolist())
    summary = precision_recall_f1(
        true_positives,
        false_positives,
        false_negatives,
    )
    summary.update(
        {
            "radius_pixels": float(radius),
            "mean_matched_distance_pixels": (
                float(np.mean(matched_distances)) if matched_distances else None
            ),
            "median_matched_distance_pixels": (
                float(np.median(matched_distances)) if matched_distances else None
            ),
        }
    )
    return summary


def count_summary(predicted_counts: Sequence[int], target_counts: Sequence[int]) -> dict[str, float]:
    predicted = np.asarray(predicted_counts, dtype=np.float64)
    target = np.asarray(target_counts, dtype=np.float64)
    if predicted.shape != target.shape or predicted.size == 0:
        raise ValueError("Count arrays must have the same non-zero shape")
    errors = predicted - target
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_predicted_count": float(np.mean(predicted)),
        "mean_target_count": float(np.mean(target)),
        "mean_signed_error": float(np.mean(errors)),
    }


def interpolated_average_precision(
    scored_predictions: Sequence[tuple[int, float, np.ndarray]],
    ground_truth: Sequence[np.ndarray],
    radius: float,
) -> dict[str, float | int]:
    """Compute confidence-ranked all-points interpolated AP at a pixel radius."""
    total_targets = sum(len(points) for points in ground_truth)
    if total_targets == 0:
        return {"average_precision": 0.0, "evaluated_predictions": 0}
    matched = [np.zeros(len(points), dtype=bool) for points in ground_truth]
    true_flags: list[float] = []
    false_flags: list[float] = []
    for image_index, _, point in sorted(
        scored_predictions,
        key=lambda item: item[1],
        reverse=True,
    ):
        targets = ground_truth[image_index]
        if len(targets) == 0:
            true_flags.append(0.0)
            false_flags.append(1.0)
            continue
        distances = np.linalg.norm(targets - point[None, :], axis=1)
        distances[matched[image_index]] = np.inf
        target_index = int(np.argmin(distances))
        if distances[target_index] <= radius:
            matched[image_index][target_index] = True
            true_flags.append(1.0)
            false_flags.append(0.0)
        else:
            true_flags.append(0.0)
            false_flags.append(1.0)
    if not true_flags:
        return {"average_precision": 0.0, "evaluated_predictions": 0}
    cumulative_true = np.cumsum(true_flags)
    cumulative_false = np.cumsum(false_flags)
    recall = cumulative_true / total_targets
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1.0)
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(
            padded_precision[index],
            padded_precision[index + 1],
        )
    changed = np.where(padded_recall[1:] != padded_recall[:-1])[0]
    average_precision = np.sum(
        (padded_recall[changed + 1] - padded_recall[changed])
        * padded_precision[changed + 1]
    )
    return {
        "average_precision": float(average_precision),
        "evaluated_predictions": len(scored_predictions),
        "maximum_recall": float(recall[-1]),
    }
