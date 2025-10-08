import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
from rich import print as rprint
from rich.progress import track

from cli_utils import resolve_exclusion_ids, resolve_input_paths
from dynamic_costaudit import compute_running_cost_and_estimate, display_cost_progress, load_pricing_config
from eval_types import (
    AnnotatedSystemResult,
    EvalContext,
    GoldStandard,
    Problem,
    Score,
    SystemUnderTest,
    assessment_from_system_result,
    problem_from_system_result,
)
from eval_utils import evaluate_feasibility, evaluate_timing, print_evaluation_summary


class EvalHarness:
    """Class containing all the information needed for a single run of a system."""

    def __init__(
        self,
        problems: List[Problem],
        system: SystemUnderTest,
        system_id: str,
        run_id: str,
        output_dir: str | PathLike,
        force_rerun: bool,
        gold_labels: Optional[List[GoldStandard]],
        show_running_costs: bool,
    ):
        """
        :param problems: The problems to evaluate the system on.
        :param system: The system under test: a function that generates an Assessment given a Problem.
        :param system_id: A unique identifier for the system under test. Usually the value of the --system CLI opt.
        :param run_id: A human-readable name for the run (for logging/caching). Use the same
            combination of run_id and output_dir to identify the same run.
        :param output_dir: The directory to write logs to for this run.
        :param force_rerun: Whether to force a rerun of all input problems, even if some input problems already have
            outputs.
        :param gold_labels: The gold labels for the input problems.
        :param show_running_costs: Whether to print the running costs of a system.
        """
        self.problems = problems
        self.system = system
        self.system_id = system_id
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.force_rerun = force_rerun

        # eval
        self.gold_labels = gold_labels
        self.show_running_costs = show_running_costs
        self.gold_by_id = {}
        if gold_labels:
            for gold in gold_labels:
                self.gold_by_id[gold.problem_id] = gold
            missing_gold_ids_or_problem_ids = set(p.problem_id for p in problems).symmetric_difference(self.gold_by_id)
            assert not missing_gold_ids_or_problem_ids

    def run_system(self, *, print_score_every=5) -> List[AnnotatedSystemResult]:
        """Run the configured system, saving assessments and other logs to the output_dir."""
        rprint(
            f"[bold yellow]Solving {len(self.problems)} {'problem' if len(self.problems) == 1 else 'problems'}[/bold"
            " yellow]"
        )
        rprint(f"System: {self.system_id}")
        rprint(f"Run ID: {self.run_id}")
        rprint(f"Running cost tracking: {self.show_running_costs}")

        # sanity checks
        if not self.gold_labels:
            rprint("[yellow]--run-eval is not set, cannot print partial scores[/yellow]")
            print_score_every = None

        # initialize cost tracking framework
        # Load pricing config for cost tracking
        pricing_config = None
        if self.show_running_costs:
            try:
                pricing_config = load_pricing_config()
                rprint("[bold green]Loaded pricing configuration for cost tracking[/bold green]")
            except Exception as e:
                rprint(f"[bold red]Could not load pricing config: {e}[/bold red]")
                self.show_running_costs = False

        # core assessment generation loop
        system_results = []
        for problem_idx, problem in enumerate(track(self.problems, description="Problems")):
            rprint(f"[bold green]===== Begin problem {problem_idx} ({problem.problem_id}) =====[/bold green]")
            try:
                result = self.run_one_problem(problem)
                # append it to the running completions
                system_results.append(result)
            except KeyboardInterrupt:
                # handle a manual interrupt
                rprint(
                    f"[bold red]Run interrupted! Completed {len(system_results)} out of"
                    f" {len(self.problems)} problems.[/bold red]"
                )
                return system_results
            except Exception as e:
                rprint(f"[bold red]{self.run_id} raised an error on problem {problem.problem_id}[/bold red]")
                traceback.print_exception(e)
                continue

            # print partial metrics (every X iterations)
            # metrics
            if print_score_every and len(system_results) > 1 and problem_idx % print_score_every == 0:
                metrics, time_metrics = self.score(system_results, ignore_missing=True, bootstrap_samples=None)
                print_evaluation_summary(metrics, time_metrics, fmt="simple")
            # cost estimation
            if self.show_running_costs and pricing_config and len(system_results) > 0:
                try:
                    running_cost, estimated_remaining = compute_running_cost_and_estimate(
                        system_results, len(self.problems), pricing_config
                    )
                    display_cost_progress(len(system_results), len(self.problems), running_cost, estimated_remaining)
                except Exception as e:
                    rprint(f"[bold red]Error computing running costs: {e}[/bold red]")

        return system_results

    def run_one_problem(self, problem: Problem) -> Optional[AnnotatedSystemResult]:
        """Run the configured system on a single Problem input, and save the assessment and logs."""
        out_dir = self.output_dir / problem.problem_id
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{self.run_id}.json"
        # check for existing
        if not self.force_rerun and out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    retval = json.load(f)
                    rprint(f"[green]Problem {problem.problem_id} found in cache, skipping[/green]")
                    return retval
            except Exception as e:
                print(f"Error loading {out_path}: {e}, rerunning the system to regenerate")

        # only give the system access to golds if it is a baseline system, just for safety
        gold_by_id = self.gold_by_id if "baseline" in self.system_id else {}
        ctx = EvalContext(
            run_id=self.run_id, output_dir=self.output_dir, force_rerun=self.force_rerun, gold_by_id=gold_by_id
        )

        # run the system with timing
        start = time.monotonic()
        result = self.system(problem, ctx)
        duration = time.monotonic() - start

        # add the extra annotations
        annotated_result: AnnotatedSystemResult = result | {
            "system_id": self.system_id,
            "run_id": self.run_id,
            "problem": problem.model_dump(),
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
        }
        # HACK: save the duration recorded by the eval harness to the Assessment object too
        annotated_result["solution"]["assessment"]["run_id"] = self.run_id
        annotated_result["solution"]["assessment"]["wall_clock_time"] = duration

        # save the assessment to a file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotated_result, f, indent=2)

        return annotated_result

    def score(
        self,
        system_results: List[AnnotatedSystemResult],
        ignore_missing: bool = True,
        bootstrap_samples=10000,
        bootstrap_ci=0.95,
    ) -> Tuple[Score, dict]:
        """
        Compute the relevant metrics for the given system results. Returns a tuple (Score, timing_dict)

        :param ignore_missing: For assessments which are missing (i.e., a gold label was provided a problem but no
            assessment was generated), skip instead of scoring it as nan.
        :param bootstrap_samples: The number of bootstrap samples to derive confidence intervals from. Can be None to
            skip bootstrap sampling.
        :param bootstrap_ci: The percentile confidence interval to calculate.
        """
        if not system_results:
            raise ValueError("Cannot compute metrics, no results given")
        # Run appropriate evaluation based on whether we were interrupted
        if ignore_missing and len(system_results) < len(self.problems):
            rprint(
                f"[bold yellow]Running partial evaluation on {len(system_results)} completed problems. Missing"
                " assessments will be ignored.[/bold yellow]"
            )
            # filter golds to only ones which have answers
            # gold_labels = [self.gold_by_id[problem_from_system_result(r).problem_id] for r in system_results]
            gold_labels = []
            for r in system_results:
                try:
                    assessment = assessment_from_system_result(r)
                    problem_id = assessment.problem_id
                    gold_labels.append(self.gold_by_id[problem_id])
                except (KeyError, AttributeError) as e:
                    rprint(f"[bold red]Could not find gold label for result: {e}[/bold red]")
                    continue
        else:
            if len(system_results) < len(self.problems):
                rprint(
                    f"[bold yellow]Running partial evaluation on {len(system_results)} completed problems. Missing"
                    " assessments will be given a score of NaN.[/bold yellow]"
                )
            else:
                rprint(f"[green]Running evaluation on {len(system_results)} completed problems.[/green]")
            gold_labels = self.gold_labels

        # sanity check to make sure all problems have a gold label
        # if bad_problem_ids := set(problem_from_system_result(r).problem_id for r in system_results).difference(
        #     set(g.problem_id for g in gold_labels)
        # ):
        if bad_problem_ids := set(assessment_from_system_result(r).problem_id for r in system_results).difference(
            set(g.problem_id for g in gold_labels)
        ):
            raise ValueError(f"Each result must have a corresponding gold label.\n{bad_problem_ids=}")
        # results_by_id = {problem_from_system_result(r).problem_id: r for r in system_results}
        results_by_id = {assessment_from_system_result(r).problem_id: r for r in system_results}

        # twiddle the data to be in the right format for evaluate_feasibility
        gold_label_values = []
        submission_values = []
        for gold_label in gold_labels:
            gold_label_values.append(gold_label.likert_score)
            system_result = results_by_id.get(gold_label.problem_id)
            if system_result:
                submission_values.append(assessment_from_system_result(system_result).likert_score)
            else:
                submission_values.append(np.nan)
        assert len(gold_label_values) == len(submission_values)

        # eval main metrics
        metrics = evaluate_feasibility(
            gold_label_values, submission_values, bootstrap_ci=bootstrap_ci, bootstrap_samples=bootstrap_samples
        )

        # eval time metrics
        time_metrics = evaluate_timing(system_results)

        return metrics, time_metrics


@click.command()
@click.option(
    "--system",
    help=(
        "Path of the system to run. This should be of the format <module>:<function> (ex:"
        " baseline_system.openai:ask_openai_with_retries)"
    ),
    required=True,
)
@click.option(
    "--problem-file",
    help=(
        "Path to the JSONL file containing input problems. For multiple input files, separate with a comma, use a glob,"
        " or pass multiple times."
    ),
    required=True,
    multiple=True,
)
@click.option(
    "--output-dir",
    help="Directory to save various run logs",
    type=click.Path(),
    required=True,
)
@click.option(
    "--run-id",
    help=(
        "Human-readable ID of the run for logging/caching (defaults to the system path). Pass the same --output-dir and"
        " --run-id to resume a previously interrupted run."
    ),
    default=None,
)
@click.option(
    "--exclude",
    help=(
        "A list of problem IDs to exclude. You may pass this option multiple times, separate IDs by commas, or use a"
        " glob-style pattern."
    ),
    multiple=True,
    default=[],
)
@click.option(
    "--force-rerun",
    help="Force re-run even if output file already exists, instead of reading output file",
    is_flag=True,
    default=False,
)
@click.option(
    "--run-eval",
    help="Run evaluation after processing problems if this is set; otherwise terminates after generating assessments",
    is_flag=True,
    default=False,
)
@click.option(
    "--gold-file",
    help=(
        "Path to the JSONL file containing gold labels for the input problems -- each input problem must have a gold"
        " label. For multiple input files, separate with a comma, use a glob, or pass multiple times."
    ),
    default=[],
    multiple=True,
)
@click.option(
    "--show-running-costs",
    help="Display running costs and estimates after each problem if set",
    is_flag=True,
    default=False,
)
def assess_problems(
    system: str,
    problem_file: List[str],
    output_dir: str,
    run_id: str,
    exclude: List[str],
    force_rerun: bool,
    run_eval: bool,
    gold_file: List[str],
    show_running_costs: bool,
):
    # resolve paths for problem_file and gold_file
    problem_paths = resolve_input_paths(problem_file)
    gold_paths = resolve_input_paths(gold_file)

    # load problems and gold (if running eval)
    problems = []
    for problem_path in problem_paths:
        with open(problem_path, encoding="utf-8") as f:
            for line in f:
                problems.append(Problem.model_validate_json(line))

    gold_labels = None
    if run_eval:
        gold_labels = []
        if not gold_paths:
            raise ValueError("A gold_file must be supplied if run_eval is True.")
        for gold_path in gold_paths:
            with open(gold_path, encoding="utf-8") as f:
                for line in f:
                    gold_labels.append(GoldStandard.model_validate_json(line))

        # validate that each problem has a gold label
        missing_gold_ids = set(p.problem_id for p in problems).difference(set(g.problem_id for g in gold_labels))
        extra_gold_ids = set(g.problem_id for g in gold_labels).difference(set(p.problem_id for p in problems))
        if missing_gold_ids:
            raise ValueError(
                "Each passed problem must have a gold_label if run_eval is True. Problems missing gold"
                f" labels:\n{missing_gold_ids}"
            )
        if extra_gold_ids:
            raise ValueError(
                "Each passed gold label should correspond to an input problem. Gold labels missing problems:"
                f"\n{extra_gold_ids}"
            )

    # deduplicate based on (claim, artifacts) - possible reuse of problem_id, possible duplicates
    # check for reused problem_ids
    seen_problem_ids = {}  # problem_id -> problem
    for problem in problems:
        if problem.problem_id in seen_problem_ids:
            # though we could disambiguate by filename, this means injecting additional context into the Problem
            # type that is filesystem-specific, and having to make sure the gold label is disambiguated the same way
            # -- for now, we'll just raise an error to prevent any future foot-guns since this doesn't happen in the
            # data we have so far.
            raise RuntimeError(f"Found a duplicate problem ID: {problem.problem_id}.")
        else:
            seen_problem_ids[problem.problem_id] = problem

    # check for identical problems
    seen_problems = []
    for problem in problems.copy():
        if (problem.claim, problem.artifacts) in seen_problems:
            rprint(f"[bold yellow]Problem {problem.problem_id} is a duplicate, discarding[/bold yellow]")
            problems.remove(problem)
            gold_labels.remove(next(g for g in gold_labels if g.problem_id == problem.problem_id))
        else:
            seen_problems.append((problem.claim, problem.artifacts))

    # TODO run N times and aggregate
    # TODO run workdir for v2

    rprint(
        f"[bold yellow]Loaded {len(problems)} {'problem' if len(problems) == 1 else 'problems'} from"
        f" {problem_paths}[/bold yellow]"
    )

    # filter out excluded problem IDs
    excluded_ids = resolve_exclusion_ids(exclude, [p.problem_id for p in problems])
    if excluded_ids:
        rprint(f"[bold yellow]Excluding {len(excluded_ids)} problems from CLI arguments:[/bold yellow]\n{excluded_ids}")
        problems = [p for p in problems if p.problem_id not in excluded_ids]
        if gold_labels is not None:
            gold_labels = [g for g in gold_labels if g.problem_id not in excluded_ids]

    # import the system under test from the CLI arg
    if ":" not in system:
        rprint(
            f"[bold red]System should be of the format <module>:<function> (ex:"
            f" baseline_system.openai:ask_openai_with_retries)[/bold red]"
        )
        sys.exit(1)
    modname, funcname = system.split(":", 1)
    # pretend we ran the script from the invoking directory; look for imports from the local dir first instead of
    # from eval_harness. We should have imported everything we need by now.
    sys.path.insert(0, ".")
    try:
        mod = importlib.import_module(modname)
        system_func = getattr(mod, funcname)
    except (ImportError, AttributeError):
        rprint(f"[bold red]Could not find system function {system}![/bold red]")
        traceback.print_exc()
        sys.exit(1)

    # build the eval harness and run
    harness = EvalHarness(
        problems=problems,
        system=system_func,
        system_id=system,
        run_id=run_id or system,
        output_dir=output_dir,
        force_rerun=force_rerun,
        gold_labels=gold_labels,
        show_running_costs=show_running_costs,
    )
    results = harness.run_system()

    rprint(f"[bold green]Finished with {len(results)}/{len(problems)} problems for {system}[/bold green]")

    # if eval, print the system eval
    if run_eval:
        metrics, time_metrics = harness.score(results)
        print_evaluation_summary(metrics, time_metrics, fmt="table")


if __name__ == "__main__":
    assess_problems()
