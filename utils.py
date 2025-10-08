from eval_harness.eval_types import Problem, Assessment, likert_to_continuous
import os
import json


def possibly_fix_json(json_output: str) -> str:
    json_output = json_output.strip()
    if "```json" in json_output:
        json_output = json_output.split("```json")[1]
        json_output = json_output.split("```")[0]
    return json_output
