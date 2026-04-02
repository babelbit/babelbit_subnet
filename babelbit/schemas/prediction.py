from pydantic import BaseModel


class BBUtteranceEvaluation(BaseModel):
    """Evaluation result for utterance prediction."""

    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class BBPredictedUtterance(BaseModel):
    index: str
    step: int
    prefix: str
    prediction: str = ""
    context: str = ""
    done: bool = False
    ground_truth: str | None = None
    evaluation: BBUtteranceEvaluation | None = None


class BBPredictOutput(BaseModel):
    success: bool
    model: str
    utterance: BBPredictedUtterance
    error: str | None = None
    context_used: str
    complete: bool = False
