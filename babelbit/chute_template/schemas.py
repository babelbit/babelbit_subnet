import os
from io import BytesIO
from typing import Any
from base64 import b64decode
from traceback import format_exc
from random import randint

from pydantic import BaseModel
# from PIL import Image

from huggingface_hub import snapshot_download
# from ultralytics import YOLO

from chutes.chute import Chute, NodeSelector
from chutes.image import Image as ChutesImage

from transformers import AutoTokenizer, AutoModelForCausalLM 


class BBUtteranceEvaluation(BaseModel):
    """Evaluation result for utterance prediction."""
    lexical_similarity: float = 0.0  
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class BBPredictedUtterance(BaseModel):
    index: str # UUID
    step: int
    prefix: str
    prediction: str = ""
    context: str = ""
    done: bool = False
    ground_truth: str | None = None  # Optional field for evaluation
    evaluation: BBUtteranceEvaluation | None = None  # Optional field for evaluation


class BBPredictOutput(BaseModel):
    success: bool
    model: str
    utterance: BBPredictedUtterance
    error: str | None = None
    context_used: str
    complete: bool = False
