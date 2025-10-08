"""
Common types used in the eval harness.
"""

import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeAlias, TypedDict

from pydantic import BaseModel, Field


# ==== input/context =====
class ArtifactType(str, Enum):
    ARTICLE = "article"
    PRESS_RELEASE = "press release"
    PAPER = "paper"
    FIGURE = "figure"
    OTHER = "other"


class Artifact(BaseModel):
    type: ArtifactType = Field(
        description="the type of artifact",
        json_schema_extra={
            "enum_descriptions": {
                "article": "a research article",
                "press release": "a press release",
                "paper": "a published research paper",
                "figure": "a figure",
                "other": "other artifact type",
            }
        },
    )
    text: str = Field(min_length=1, description="text of the artifact where the claim is made")


class Problem(BaseModel):
    type: str = "problem"
    format_version: str = "1.0"
    problem_id: str
    problem_version: str
    domain: str
    subdomain: Optional[str] = None
    claim: str = Field(min_length=1, description="the claim to be assessed")
    artifacts: List[Artifact] = Field(
        description="a list of artifacts where the claim is made",
    )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)


class EvidenceType(str, Enum):
    CLAIM = "claim"
    PARAMETRIC_KNOWLEDGE = "parametric knowledge"
    ARTIFACT_KNOWLEDGE = "artifact knowledge"
    WEB_SEARCH = "web search"
    KB_KNOWLEDGE = "knowledge base search"
    LOCAL_FILE_SEARCH = "local file search"
    SIMULATION_RESULT = "simulation result"
    REASONING = "reasoning"
    LLM_ASSESSMENT = "llm_assessment" 
    OTHER = "other"


class EvidenceItem(BaseModel):
    type: EvidenceType = Field(
        description="the type of evidence",
        json_schema_extra={
            "enum_descriptions": {
                "claim": "evidence derived from the statement of the claim",
                "parametric knowledge": "evidence derived from parametric knowledge of the LLM",
                "artifact knowledge": "evidence derived from the artifact text",
                "web search": "evidence derived from a web search result",
                "knowledge base search": (
                    "evidence derived from a knowledge base items presented in the context of the LLM call"
                ),
                "local file search": "evidence derived from a local file search result",
                "simulation result": "evidence derived from a simulation result",
                "reasoning": "evidence derived from a previous reasoning output of the LLM",
                "llm_assessment": "evidence derived from an LLM assessment",
                "other": "other evidence type",
            }
        },
    )
    id: str = Field(min_length=1, description="a unique identifier for the evidence")
    source: Optional[str] = Field(
        min_length=1,
        description="the source of the evidence, e.g. a URL or a file name",
    )
    citation: Optional[str] = Field(
        default=None,
        description=(
            "one or more snippets from the source supporting the assessment step. Multiple snippets are seperated"
            " by ..."
        ),
    )


class ExplanationItem(BaseModel):
    text: str = Field(min_length=1)
    evidence: List[str] = Field(
        min_length=0, #for testing with codescientist which doesnt have evidence rightnow
        description="a list of evidence identifiers",
    )


def likert_to_continuous(likert_score: int) -> float:
    likert_score = max(-2, min(2, likert_score))  # clip to [-2, 2]
    # return (likert_score + 2) / 4  # linear mapping
    score = 1 / (1 + math.exp(-2.5 * likert_score))  # sigmoid mapping
    return round(score, 2)


class Assessment(BaseModel):
    type: str = "assessment"
    format_version: str = "0.2"
    problem_id: str
    problem_version: Optional[str] = None # made optional temporarily for codescientist
    team: str = "upenn"
    run_id: str = ""
    likert_score: int = Field(
        description="A score between -2 (extremely infeasible) and +2 (extremely feasible)",
        ge=-2,
        le=2,
    )
    continuous_score: Optional[float] = Field(  # made optional temporarily for codescientist
        description="likert_score mapped to a [0, 1] interval",
        ge=0,
        le=1,
    )
    confidence: Optional[float] = Field( # made optional temporarily for codescientist
        description="a value in [0, 1] reflecting the confidence in the assessment.",
        ge=0,
        le=1,
    )
    wall_clock_time: Optional[float] = 0.0
    explanation: List[ExplanationItem] = Field(
        min_length=1,
        description="a list of explanations for the assessment",
    )
    evidence: Dict[str, EvidenceItem] = Field(
        description="a dictionary of evidence items",
    )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def to_apl_json(self) -> str:
        apl_dict = self.model_dump()
        for key, value in apl_dict["evidence"].items():
            del value["id"]
            apl_dict["evidence"][key] = value
        return json.dumps(apl_dict)


class GoldStandard(BaseModel):
    """Pydantic model for loading released gold label data"""

    type: str = "gold standard"
    format_version: str = "1.0"
    problem_id: str
    problem_version: str
    domain: str
    subdomain: str
    claim: str
    likert_score: int
    explanation: Any
    evidence: dict
    author: str
    tags: Dict[str, Any]
    comments: List[str]


class EvalContext(BaseModel):
    run_id: str
    output_dir: Path
    force_rerun: bool
    # only provided to baseline systems (if "baseline" in system path)
    gold_by_id: dict[str, GoldStandard]


# ==== output ====
class SystemResultSolution(TypedDict):
    assessment: dict  # dict of an Assessment object, the only required key
    # some legacy systems return extra keys; systems may save whatever other data they want here
    # thought: str
    # tool_call_info: str


class SystemResult(TypedDict):
    """
    The expected return dict shape from a system under test.

    The system MUST return at least a solution and a list of usage objects, but MAY return any extra JSON-serializable
    data it wants to persist to disk.

    Certain keys will be overwritten by the eval harness (see AnnotatedSystemResult below).
    """

    solution: SystemResultSolution
    usage: Any  # TODO this should be standardized, rn can be list|dict
    # systems may save extra JSON-serializable data in keys here too


class AnnotatedSystemResult(SystemResult):
    """Additional keys added to each system result by the eval harness. Will overwrite these keys if present."""

    run_id: str
    problem: dict  # the input problem in dict format
    timestamp: str  # default: datetime.now().isoformat()
    duration: float  # in seconds


def problem_from_system_result(result: AnnotatedSystemResult) -> Problem:
    problem_data = result["problem"]

    if isinstance(problem_data, str):
        problem_data = json.loads(problem_data)

    return Problem.model_validate(problem_data)


def assessment_from_system_result(result: SystemResult) -> Assessment:
    try:
        assessment_data = result["solution"]["assessment"]
    except KeyError as e:
        assessment_data = result["solution"]["json_output"]  # legacy key

    if isinstance(assessment_data, str):
        assessment_data = json.loads(assessment_data)

    return Assessment.model_validate(assessment_data)


class ScoreStats(BaseModel):
    """Statistics for a single metric."""

    observed: float
    # bootstrap config
    bootstrap_samples: Optional[int]
    bootstrap_ci: Optional[float]
    # bootstrap sampled metrics
    mean: Optional[float]
    stderr: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]


class Score(BaseModel):
    """Container for the statistical metrics to evaluate a run."""

    # bootstrap config - should be the same for them all
    bootstrap_samples: Optional[int]
    bootstrap_ci: Optional[float]
    # metrics
    quadratic_cohen_kappa: ScoreStats
    krippendorff_alpha: ScoreStats
    pearson: ScoreStats
    mean_absolute_error: ScoreStats
    accuracy: ScoreStats
    balanced_accuracy: ScoreStats
    sign_accuracy: ScoreStats


# ==== helpers =====
SystemUnderTest: TypeAlias = Callable[[Problem, EvalContext], SystemResult]
"""The system being evaluated. Given a Problem instance, it returns its assessment + metadata in a SystemResult dict."""
