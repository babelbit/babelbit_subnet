import json

from babelbit.utils.file_handling import get_processed_miners_for_challenge


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_processed_miners_filters_by_challenge_type(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()

    challenge_uid = "challenge-abc"

    # Legacy untyped file (treated as main)
    _write_json(
        scores_dir / "dialogue_run_challenge-abc_miner_1_run.json",
        {
            "challenge_uid": challenge_uid,
            "miner_uid": 1,
            "miner_hotkey": "hk1",
        },
    )

    # Typed main file
    _write_json(
        scores_dir / "dialogue_run_challenge-abc_type_main_miner_2_run.json",
        {
            "challenge_uid": challenge_uid,
            "challenge_type": "main",
            "miner_uid": 2,
            "miner_hotkey": "hk2",
        },
    )

    # Legacy typed round2 file (should map to arena)
    _write_json(
        scores_dir / "dialogue_run_challenge-abc_type_round2_miner_3_run.json",
        {
            "challenge_uid": challenge_uid,
            "challenge_type": "round2",
            "miner_uid": 3,
            "miner_hotkey": "hk3",
        },
    )

    # Typed arena file
    _write_json(
        scores_dir / "dialogue_run_challenge-abc_type_arena_miner_4_run.json",
        {
            "challenge_uid": challenge_uid,
            "challenge_type": "arena",
            "miner_uid": 4,
            "miner_hotkey": "hk4",
        },
    )

    main_processed = get_processed_miners_for_challenge(str(scores_dir), challenge_uid, challenge_type="main")
    arena_processed = get_processed_miners_for_challenge(str(scores_dir), challenge_uid, challenge_type="arena")

    assert main_processed == {(1, "hk1"), (2, "hk2")}
    assert arena_processed == {(3, "hk3"), (4, "hk4")}
