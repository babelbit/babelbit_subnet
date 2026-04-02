from pathlib import Path

from babelbit.utils.file_handling import (
    create_challenge_summary_data,
    normalize_challenge_type,
    save_challenge_summary_file,
    save_dialogue_score_file,
)
from babelbit.utils.miner_registry import Miner


def test_challenge_type_normalization():
    assert normalize_challenge_type("main") == "main"
    assert normalize_challenge_type("solo") == "solo"
    assert normalize_challenge_type("arena") == "arena"
    assert normalize_challenge_type("round2") == "arena"
    assert normalize_challenge_type("round1") == "main"


def test_dialogue_score_filename_uses_challenge_type(tmp_path):
    score_data = {
        "challenge_uid": "ch-1",
        "challenge_type": "solo",
        "dialogue_uid": "dlg-1",
        "miner_uid": 8,
        "miner_hotkey": "hk8",
        "utterances": [],
        "dialogue_summary": {"average_U_best_early": 0.5},
    }

    filepath = save_dialogue_score_file(score_data, output_dir=str(tmp_path))
    name = Path(filepath).name

    assert "_type_solo_" in name
    assert "_round_" not in name


def test_challenge_summary_filename_uses_arena_type(tmp_path):
    summary_data = {
        "challenge_uid": "ch-2",
        "challenge_type": "round2",
        "miner_uid": 9,
        "miner_hotkey": "hk9",
        "dialogues": [],
        "challenge_mean_U": 0.7,
    }

    filepath = save_challenge_summary_file(summary_data, output_dir=str(tmp_path))
    name = Path(filepath).name

    assert "_type_arena_" in name
    assert "_round_" not in name


def test_create_challenge_summary_data_sets_challenge_type():
    miner = Miner(uid=1, hotkey="hk1", block=10)
    summary = create_challenge_summary_data(
        challenge_uid="ch-3",
        miner=miner,
        dialogue_scores=[0.2],
        dialogue_uids=["dlg-1"],
        challenge_type="solo",
    )

    assert summary["challenge_type"] == "solo"
    assert "challenge_round" not in summary
