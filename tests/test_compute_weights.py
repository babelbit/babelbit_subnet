import pytest

from babelbit.cli.validate import compute_weights
from babelbit.cli import validate as validate_mod


def test_compute_weights_with_trailing_scores():
    winner_uid = 10
    trailing = {2: 3.0, 5: 1.0}  # total trailing = 4.0

    weights, uids = compute_weights(winner_uid, trailing)

    assert uids == [10, 2, 5]
    assert weights == pytest.approx([0.95, 0.0375, 0.0125])
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0


def test_compute_weights_no_trailing_scores():
    weights, uids = compute_weights(7, {})

    assert uids == [7]
    assert weights == [1.0]


def test_compute_weights_zero_or_negative_trailing():
    # Negative/None/zero trailing scores are clamped to zero and ignored in distribution
    trailing = {3: -1.0, 4: 0.0, 8: None}

    weights, uids = compute_weights(1, trailing)

    assert uids == [1]
    assert weights == [1.0]


def test_compute_weights_close_to_winner():
    winner_uid = 20
    trailing = {6: 94.0, 9: 5.0}  # total trailing = 99.0

    weights, uids = compute_weights(winner_uid, trailing)

    assert uids == [20, 6, 9]
    # Winner gets 95%; trailing miners split the remaining 5% in proportion to their scores
    assert pytest.approx(weights[0]) == 0.95
    trailing_weights = weights[1:]
    assert pytest.approx(sum(trailing_weights)) == 0.05
    # Ratio of trailing weights should match ratio of trailing scores
    assert pytest.approx(trailing_weights[0] / trailing_weights[1]) == pytest.approx(trailing[6] / trailing[9])


def test_compute_weights_splits_main_and_arena_winners():
    trailing = {3: 2.0, 4: 1.0}
    weights, uids = compute_weights(
        1,
        trailing,
        arena_winner_uid=2,
        arena_incentive_fraction=0.4,
    )

    assert uids == [1, 2, 3, 4]
    assert weights[0] == pytest.approx(0.57)  # 95% core * 60% main
    assert weights[1] == pytest.approx(0.38)  # 95% core * 40% arena
    assert weights[2] == pytest.approx((2.0 / 3.0) * 0.05)
    assert weights[3] == pytest.approx((1.0 / 3.0) * 0.05)
    assert sum(weights) == pytest.approx(1.0)


def test_compute_weights_combines_main_and_arena_when_same_winner():
    weights, uids = compute_weights(
        7,
        {},
        arena_winner_uid=7,
        arena_incentive_fraction=0.25,
    )

    assert uids == [7]
    assert weights == [1.0]


def test_compute_weights_routes_missing_arena_share_to_burn_uid():
    weights, uids = compute_weights(
        7,
        {},
        arena_winner_uid=None,
        arena_incentive_fraction=0.4,
        burn_uid=248,
    )

    assert uids == [7, 248]
    assert weights == pytest.approx([0.6, 0.4])


def test_normalize_weight_vector_renormalizes_malformed_sum():
    weights = validate_mod._normalize_weight_vector([0.7, 0.7])

    assert weights == pytest.approx([0.5, 0.5])
    assert sum(weights) == pytest.approx(1.0)


def test_normalize_weight_vector_drops_non_positive_total():
    assert validate_mod._normalize_weight_vector([0.0, 0.0]) == []
