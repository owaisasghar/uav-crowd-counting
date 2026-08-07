import numpy as np

from scripts.p2p_metrics import (
    count_summary,
    interpolated_average_precision,
    localization_summary,
    match_point_sets,
)


def test_match_point_sets_is_one_to_one_and_radius_gated():
    predicted = np.array([[0, 0], [1, 0], [20, 20]], dtype=np.float32)
    target = np.array([[0, 0], [10, 10]], dtype=np.float32)
    distances = match_point_sets(predicted, target, radius=2)
    assert len(distances) == 1
    assert distances[0] == 0


def test_localization_summary_counts_tp_fp_fn():
    result = localization_summary(
        [np.array([[0, 0], [50, 50]], dtype=np.float32)],
        [np.array([[1, 0], [10, 10]], dtype=np.float32)],
        radius=2,
    )
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_count_summary():
    result = count_summary([8, 12], [10, 10])
    assert result["mae"] == 2
    assert result["rmse"] == 2
    assert result["mean_signed_error"] == 0


def test_average_precision_is_one_for_ranked_exact_matches():
    ground_truth = [np.array([[0, 0], [10, 10]], dtype=np.float32)]
    predictions = [
        (0, 0.9, np.array([0, 0], dtype=np.float32)),
        (0, 0.8, np.array([10, 10], dtype=np.float32)),
        (0, 0.1, np.array([50, 50], dtype=np.float32)),
    ]
    result = interpolated_average_precision(predictions, ground_truth, radius=1)
    assert result["average_precision"] == 1
    assert result["maximum_recall"] == 1
