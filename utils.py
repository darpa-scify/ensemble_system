from openai import OpenAI
from dotenv import load_dotenv
from scifytypes import Problem, Assessment, likert_to_continuous
import os
import json
import config

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def possibly_fix_json(json_output: str) -> str:
    json_output = json_output.strip()
    if "```json" in json_output:
        json_output = json_output.split("```json")[1]
        json_output = json_output.split("```")[0]
    return json_output


def md_to_json(problem: Problem, markdown: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": open(
                            "prompts/OAI_MD_TO_JSON_PROMPT.md", "r", encoding="utf-8"
                        ).read(),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Problem:\n{problem.to_json()}\n\nAssessment Markdown:\n{markdown}",
                    }
                ],
            },
        ],
        response_format={"type": "text"},
        temperature=1,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content, response.usage


def md_to_json_with_retries(problem: Problem, markdown: str, retries: int = 3):
    for _ in range(retries):
        try:
            response, usage = md_to_json(problem, markdown)
            json_output = possibly_fix_json(response)
            response = json.loads(json_output)
            assessment = Assessment.model_validate(response)
            assessment.problem_id = problem.problem_id
            assessment.problem_version = problem.problem_version
            assessment.team = config.TEAM
            assessment.format_version = problem.format_version
            assessment.continuous_score = likert_to_continuous(assessment.likert_score)
            return assessment.to_json(), usage
        except Exception as e:
            print(f"Error converting markdown to JSON: {e}")
            continue
    raise Exception(f"Failed to convert markdown to JSON after {retries} retries")
