import json
import os
import random
from time import sleep
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
import traceback
# from scifytypes import Assessment, Problem
from eval_harness.eval_types import Assessment, Problem
from ensemble_system.utils import get_assessment
from utils import possibly_fix_json
from rich import print as rprint
# from scifytypes import likert_to_continuous
from eval_harness.eval_types import likert_to_continuous

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model_name = "claude-sonnet-4-5-20250929"
SYSTEM_PROMPT = (
    open("prompts/BASELINE_ENSEMBLE_PROMPT_IMPROVED_V2.md", "r", encoding="utf-8").read().strip()
)


def ensemble(run_id: str, problem: Problem, assessments: list[Assessment]):
    if not assessments:
        raise ValueError("No assessments to ensemble")

    if len(assessments) == 1:
        rprint(
            f"{problem.problem_id} has only one assessment; setting ensemble value to the single assessment"
        )
        ensemble_assessment = assessments[0].model_copy()
        ensemble_assessment.run_id = run_id
        return {
            "run_id": run_id,
            "problem": problem.to_json(),
            "usage": {},
            "duration": 0,
            "timestamp": datetime.now().isoformat(),
            "agent_config": {
                "model": model_name,
                "max_tokens": 10000,
                "temperature": 1,
            },
            "solution": {
                "thought": "",
                "tool_call_info": "",
                "json_output": ensemble_assessment.to_json(),
            },
        }

    input_jsons = "\n\n".join([assessment.to_json() for assessment in assessments])
    assessment_schema = json.dumps(Assessment.model_json_schema(), indent=2)
    input_message = f"""Assessment Schema (for reference - each assessment below follows this structure):
{assessment_schema}

- Assessments to ensemble :
{input_jsons}"""

    rprint(f"Ensembling {len(assessments)} assessments")
    response = client.with_options(timeout=600).messages.create(
        model=model_name,
        max_tokens=10000,
        temperature=1,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": input_message}],
            }
        ],
        thinking={"type": "enabled", "budget_tokens": 4096},
    )
    usage = response.usage
    thought = response.content[0].thinking.strip()
    json_output = possibly_fix_json(response.content[-1].text)

    response_dict = {
        "usage": usage.to_dict(),
        "agent_config": {
            "model": model_name,
            "max_tokens": 10000,
            "temperature": 1,
        },
        "solution": {
            "json_output": json_output,
            "thought": thought,
        },
    }

    assessment = get_assessment(response_dict, redirects=True)
    response_dict["solution"]["assessment"] = assessment.model_dump()
    return response_dict



def ensemble_with_retries(
    run_id: str, problem: Problem, assessments: list[Assessment], retries: int = 3
):
    for _ in range(retries):
        try:
            response = ensemble(run_id, problem, assessments)
            response["run_id"] = run_id
            assessment = get_assessment(response, redirects=True)
            assessment.run_id = run_id
            assessment.problem_id = problem.problem_id
            assessment.problem_version = problem.problem_version
            assessment.team = "upenn"
            assessment.format_version = problem.format_version
            assessment.continuous_score = likert_to_continuous(assessment.likert_score)
            response["solution"]["assessment"] = assessment.model_dump()
            return response
        except Exception as e:
            rprint(f"Error ensembling {problem.problem_id}: {e}. Retrying...")
            traceback.print_exc()
            sleep(random.randint(2, 5))
    raise Exception(f"Failed to ensemble {problem.problem_id} after {retries} retries")
