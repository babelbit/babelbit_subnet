import pytest

from babelbit.cli.validate import compute_weights
from babelbit.cli import validate as validate_mod


def test_compute_weights_linear_main_distribution():
    weights, uids = compute_weights(main_uid_scores={10: 3.0, 2: 1.0})

    assert uids == [10, 2]
    assert weights == pytest.approx([0.75, 0.25])
    assert sum(weights) == pytest.approx(1.0)


def test_compute_weights_ignores_non_positive_main_scores():
    weights, uids = compute_weights(
        main_uid_scores={3: -1.0, 4: 0.0, 8: None},  # type: ignore[arg-type]
        burn_uid=248,
    )

    assert uids == [248]
    assert weights == [1.0]


def test_compute_weights_splits_linear_main_and_arena_winner():
    weights, uids = compute_weights(
        main_uid_scores={1: 0.9, 3: 0.2, 4: 0.5},
        arena_winner_uid=2,
        arena_incentive_fraction=0.4,
    )

    # Main split (60%) linear by main scores:
    # uid1=0.3375, uid3=0.075, uid4=0.1875; plus arena winner uid2=0.4
    assert uids == [1, 3, 4, 2]
    assert weights == pytest.approx([0.3375, 0.075, 0.1875, 0.4])
    assert sum(weights) == pytest.approx(1.0)


def test_compute_weights_combines_main_and_arena_when_same_uid():
    weights, uids = compute_weights(
        main_uid_scores={7: 1.0, 8: 1.0},
        arena_winner_uid=7,
        arena_incentive_fraction=0.25,
    )

    # Main split 75% linearly => 37.5% each, arena adds 25% to uid7.
    assert uids == [7, 8]
    assert weights == pytest.approx([0.625, 0.375])


def test_compute_weights_routes_missing_arena_share_to_burn_uid():
    weights, uids = compute_weights(
        main_uid_scores={7: 1.0},
        arena_winner_uid=None,
        arena_incentive_fraction=0.4,
        burn_uid=248,
    )

    assert uids == [7, 248]
    assert weights == pytest.approx([0.6, 0.4])


def test_compute_weights_routes_full_arena_share_to_winner():
    weights, uids = compute_weights(
        main_uid_scores={7: 1.0, 9: 3.0},
        arena_winner_uid=42,
        arena_incentive_fraction=1.0,
        burn_uid=248,
    )

    assert uids == [42]
    assert weights == [1.0]


def test_compute_weights_routes_full_arena_share_to_burn_without_winner():
    weights, uids = compute_weights(
        main_uid_scores={7: 1.0, 9: 3.0},
        arena_winner_uid=None,
        arena_incentive_fraction=1.0,
        burn_uid=248,
    )

    assert uids == [248]
    assert weights == [1.0]


def test_normalize_weight_vector_renormalizes_malformed_sum():
    weights = validate_mod._normalize_weight_vector([0.7, 0.7])

    assert weights == pytest.approx([0.5, 0.5])
    assert sum(weights) == pytest.approx(1.0)


def test_normalize_weight_vector_drops_non_positive_total():
    assert validate_mod._normalize_weight_vector([0.0, 0.0]) == []
