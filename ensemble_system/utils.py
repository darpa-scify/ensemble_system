import json
import re
from typing import Dict

import requests
from rich import print as rprint

from eval_harness.eval_types import Assessment


def resolve_redirects(url: str) -> str:
    response = requests.head(url, timeout=10, allow_redirects=True)
    return response.url


def extract_final_json_block(text: str) -> str:
    stack = []
    start_idx = None
    last_valid_candidate = None

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        json.loads(candidate)  # confirm it's valid
                        last_valid_candidate = candidate  # update final valid one
                    except Exception:
                        pass  # skip malformed blocks

    return last_valid_candidate.strip() if last_valid_candidate else None


def possibly_fix_json(json_output: str) -> str:
    json_output = json_output.strip()

    # Extract JSON from markdown code blocks
    if "```json" in json_output:
        json_output = re.search(r"```json(.*?)```", json_output, re.DOTALL).group(1)

    # Fix malformed JSON
    json_output = extract_final_json_block(json_output)

    return json_output.strip()


def get_assessment(response_dict: Dict, redirects: bool = False) -> Assessment:
    json_output = response_dict["solution"]["json_output"]
    # print(json_output)
    json_dict = json.loads(json_output)
    try:
        if redirects:
            for ek, ev in json_dict["evidence"].items():
                if ev["type"] == "web search":
                    json_dict["evidence"][ek]["source"] = resolve_redirects(
                        ev["source"].replace("?utm_source=openai", "")
                    )
    except Exception as e:
        rprint(f"Error: ek: {ek}, ev: {ev}")
    return Assessment.model_validate(json_dict)
