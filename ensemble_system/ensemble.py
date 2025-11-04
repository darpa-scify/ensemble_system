from eval_harness.eval_types import EvalContext, SystemResult
import json
from pathlib import Path
from rich import print as rprint
from ensemble_system.ensemble_utils import ensemble_with_retries
from eval_harness.eval_types import Assessment, Problem

def baseline_ensemble(problem: Problem, ctx: EvalContext, num_attempts: int = 3) -> SystemResult:
    """Run all the other baseline systems, then ask Claude to merge them"""
    subsystem_configs = {
        "ScholarQA": ("runs/eval/sprint1_eval_scholarqa_assessments_all", "scholar_qa", "scholarqa"),
        "Claimspy_V2": ("runs/eval/claimspy_v2_eval_all-internal-latest", "claimspy_V2", "claimspy_v2"),
    }
    
    assessments = []
    
    # Read assessments from directories
    for subsystem, (base_dir, fname, run_id) in subsystem_configs.items():
        try:
            # Construct path: base_dir/problem_id/run_id.json
            eval_file = Path(base_dir) / problem.problem_id / f"{fname}.json"
            # print(f"Reading assessment from {eval_file}")
            
            if not eval_file.exists():
                print(f"File not found: {eval_file}")
                raise FileNotFoundError
            
            # Read the JSON file
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # print(type(data))
            # Extract assessment from ["solution"]["assessment"]
            assessment = data.get("solution").get("assessment")
            assessment["run_id"] = run_id  # Add run_id to assessment for tracking
            
            if assessment is None:
                print(f"No assessment found in {eval_file}")
                raise Exception(f"No assessment found in {base_dir} (run: {run_id}) for Problem : { problem.problem_id}")
            
            assessment["run_id"] = run_id 
            assessments.append(assessment)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {eval_file}: {e}")
            continue
        except Exception as e:
            print(f"Error reading assessment for {subsystem} (run: {run_id}) for {problem.problem_id}: {e}")
            continue

    rprint(f"Ensembling {problem.problem_id}")
    parsed_assessments = []
    for assessment in assessments:
        try:
            parsed_assessments.append(Assessment.model_validate(assessment))
        except Exception as e:
            rprint(f"Error parsing assessment {assessment['run_id']}: {e}. Skipping")
            continue

    return ensemble_with_retries(ctx.run_id, problem, parsed_assessments)


