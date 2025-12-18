import os
import json
import random
import asyncio
from datetime import datetime
import time
from logging import getLogger
from typing import List, Optional, Set, Tuple
from babelbit.utils.miner_registry import get_miners_from_registry, Miner
from babelbit.chute_template.schemas import BBPredictedUtterance

logger = getLogger(__name__)

def create_dialogue_score_file_data(miner: Miner, challenge_uid: Optional[str], dialogue_uid: str, evaluated_utterances: List[BBPredictedUtterance], dialogue_score: float, log_file_path: Optional[str] = None):
    """
    Create score file data structure matching the expected JSON format.
    """
    # Group utterances by utterance number for the output format
    utterances_by_number = {}
    for utterance in evaluated_utterances:
        utterance_num = getattr(utterance, 'utterance_number', 1)  # Default to 1 if not set
        if utterance_num not in utterances_by_number:
            utterances_by_number[utterance_num] = {
                'utterance_number': utterance_num,
                'ground_truth': utterance.ground_truth or "",
                'steps': [],
                'best_step': 0,
                'U_best': -1.0,
                'total_steps': 0
            }
        
        # Add this step to the utterance
        if utterance.evaluation:
            step_data = {
                'step': len(utterances_by_number[utterance_num]['steps']),
                'lexical_similarity': getattr(utterance.evaluation, 'lexical_similarity', 0.0),
                'semantic_similarity': getattr(utterance.evaluation, 'semantic_similarity', 0.0),
                'earliness': getattr(utterance.evaluation, 'earliness', 0.0),
                'U_step': getattr(utterance.evaluation, 'u_step', 0.0),
                'prefix': utterance.prediction or ""
            }
            utterances_by_number[utterance_num]['steps'].append(step_data)
            
            # Update best step if this one is better
            if step_data['U_step'] > utterances_by_number[utterance_num]['U_best']:
                utterances_by_number[utterance_num]['U_best'] = step_data['U_step']
                utterances_by_number[utterance_num]['best_step'] = step_data['step']
    
    # Update total steps count
    for utterance_data in utterances_by_number.values():
        utterance_data['total_steps'] = len(utterance_data['steps'])
    
    return {
        'log_file': log_file_path,
        'challenge_uid': challenge_uid,
        'dialogue_uid': dialogue_uid,
        'miner_uid': miner.uid,
        'miner_hotkey': miner.hotkey,
        'utterances': list(utterances_by_number.values()),
        'dialogue_summary': {
            'average_U_best_early': dialogue_score
        }
    }


def save_dialogue_score_file(score_data: dict, output_dir: str = "scores"):
    """
    Save score data to JSON file with timestamp.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    miner_uid = score_data.get('miner_uid', 'unknown')
    dialogue_uid = score_data.get('dialogue_uid', 'unknown')
    challenge_uid = score_data.get('challenge_uid', 'unknown')

    filename = f"dialogue_run_{challenge_uid}_miner_{miner_uid}_dlg_{dialogue_uid}_run_{timestamp}-score.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(score_data, f, indent=2)
    
    logger.info(f"Saved score file: {filepath}")
    return filepath


def aggregate_utterance_scores_to_dialogue(utterance_scores: List[float]) -> float:
    """
    Aggregate utterance scores into a single dialogue score.
    Dialogue score is just the average of the utterance scores.
    """
    if not utterance_scores:
        return 0.0
    
    return sum(utterance_scores) / len(utterance_scores)


def create_challenge_summary_data(
    challenge_uid: str,
    miner: Miner,
    dialogue_scores: List[float],
    dialogue_uids: List[str],
    run_file_path: Optional[str] = None,
    challenge_file_path: Optional[str] = None,
    challenge_type: str = "main",
) -> dict:
    """
    Create challenge summary data structure for the overall challenge results.
    """
    dialogues = []
    for i, (dialogue_uid, score) in enumerate(zip(dialogue_uids, dialogue_scores)):
        dialogues.append({
            "dialogue_average_u_best_early": score,
            "dialogue_uid": dialogue_uid,
            "dialogue_index": i
        })
    
    # Calculate challenge mean score
    challenge_mean_u = sum(dialogue_scores) / len(dialogue_scores) if dialogue_scores else 0.0
    
    return {
        "run_file": run_file_path,
        "challenge_file": challenge_file_path,
        "challenge_uid": challenge_uid,
        "challenge_type": challenge_type,
        "miner_uid": miner.uid,
        "miner_hotkey": miner.hotkey,
        "dialogues": dialogues,
        "challenge_mean_U": challenge_mean_u
    }


def save_challenge_summary_file(summary_data: dict, output_dir: str = "scores"):
    """
    Save challenge summary data to JSON file with timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    challenge_uid = summary_data.get('challenge_uid', 'unknown')
    miner_uid = summary_data.get('miner_uid', 'unknown')
    
    filename = f"challenge_run_{challenge_uid}_miner_{miner_uid}_run_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Saved challenge summary file: {filepath}")
    return filepath


def get_processed_miners_for_challenge(output_dir: str, challenge_uid: str) -> Set[Tuple[int, str]]:
    """Scan existing score files in output_dir and return set of (miner_uid, miner_hotkey)
    that have already produced results for the given challenge_uid.

    A miner is considered processed if any JSON file in the directory contains matching
    challenge_uid, miner_uid and miner_hotkey fields. Both dialogue and challenge summary
    files qualify.
    """
    processed: Set[Tuple[int, str]] = set()
    if not os.path.isdir(output_dir):
        return processed

    for fname in os.listdir(output_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and data.get('challenge_uid') == challenge_uid
                and 'miner_uid' in data
                and 'miner_hotkey' in data
            ):
                try:
                    miner_uid = int(data['miner_uid'])
                    miner_hotkey = str(data['miner_hotkey'])
                    processed.add((miner_uid, miner_hotkey))
                except Exception:
                    continue
        except Exception:
            # Ignore unreadable or malformed files
            continue
    if processed:
        logger.info(f"Detected {len(processed)} already processed miners for challenge {challenge_uid}")
    return processed
