import json
import os
from glob import glob
from typing import Any, Dict, List, Tuple

import pandas as pd
from rich import print as rprint

from eval_types import AnnotatedSystemResult, Problem


def load_pricing_config(config_path: str = "eval/pricing_config.json") -> Dict[str, Any]:
    """Load pricing configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)


def get_model_pricing(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pricing for a specific model"""
    # Direct match
    if model_name in config:
        return config[model_name]

    # Fuzzy matching for model names
    for model_key in config.keys():
        if model_key in model_name or model_name in model_key:
            return config[model_key]

    raise ValueError(f"Pricing not found for model: {model_name}")


def normalize_usage_gemini_dynamic(usage: dict, config: Dict[str, Any]) -> dict:
    """Dynamic Gemini pricing based on config"""
    model_name = usage["model"]
    pricing = get_model_pricing(model_name, config)

    result = {}
    input_token_count = usage["prompt_token_count"]
    total_token_count = usage["total_token_count"]
    output_token_count = total_token_count - input_token_count
    num_web_searches = usage.get("web_search_requests", 0)

    result["input_token_count"] = input_token_count
    result["output_token_count"] = output_token_count

    threshold = pricing.get("threshold", 200000)
    input_price = (
        pricing["input_price_per_token_over_200k"]
        if input_token_count > threshold
        else pricing["input_price_per_token_under_200k"]
    )
    output_price = (
        pricing["output_price_per_token_over_200k"]
        if output_token_count > threshold
        else pricing["output_price_per_token_under_200k"]
    )

    result["caching_used"] = False
    result["cost"] = (
        input_token_count * input_price + output_token_count * output_price
    ) / 1000000 + num_web_searches * pricing.get("grounding_search_price_per_1000", 35.0) / 1000
    result["cost_without_caching"] = result["cost"]

    return result


def normalize_usage_gpt_dynamic(usage: dict, config: Dict[str, Any]) -> dict:
    """Dynamic OpenAI pricing based on config"""
    model_name = usage["model"]
    pricing = get_model_pricing(model_name, config)

    result = {}
    cached_tokens = 0
    num_web_searches = usage.get("web_search_requests", 0)

    if "prompt_tokens" in usage:
        input_token_count = usage["prompt_tokens"]
        output_token_count = usage["completion_tokens"]
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"]["cached_tokens"]:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
    else:
        input_token_count = usage["input_tokens"]
        output_token_count = usage["output_tokens"]
        if "input_tokens_details" in usage and usage["input_tokens_details"]["cached_tokens"]:
            cached_tokens = usage["input_tokens_details"]["cached_tokens"]

    result["input_token_count"] = input_token_count
    result["output_token_count"] = output_token_count
    result["caching_used"] = cached_tokens > 0

    # Note: OpenAI doesn't have built-in web search, so using external service cost
    web_search_cost_per_request = pricing.get("web_search_cost_per_request", 0.01)

    result["cost"] = (
        pricing["cached_price_per_token"] * cached_tokens
        + pricing["input_price_per_token"] * (input_token_count - cached_tokens)
        + pricing["output_price_per_token"] * output_token_count
    ) / 1000000 + num_web_searches * web_search_cost_per_request

    result["cost_without_caching"] = (
        pricing["input_price_per_token"] * input_token_count + pricing["output_price_per_token"] * output_token_count
    ) / 1000000 + num_web_searches * web_search_cost_per_request

    return result


def normalize_usage_claude_dynamic(usage: dict, config: Dict[str, Any]) -> dict:
    """Dynamic Anthropic pricing based on config"""
    model_name = usage["model"]
    pricing = get_model_pricing(model_name, config)

    result = {}
    input_token_count = usage["input_tokens"]
    output_token_count = usage["output_tokens"]
    cache_write_tokens = usage["cache_creation_input_tokens"]
    cache_read_tokens = usage["cache_read_input_tokens"]
    num_web_searches = usage.get("web_search_requests", 0)

    result["input_token_count"] = input_token_count
    result["output_token_count"] = output_token_count
    result["caching_used"] = (cache_write_tokens + cache_read_tokens) > 0

    # Claude web search cost (built-in feature)
    web_search_cost_per_request = pricing.get("web_search_cost_per_request", 0.01)

    result["cost"] = (
        pricing["cache_write_price_per_token"] * cache_write_tokens
        + pricing["cache_read_price_per_token"] * cache_read_tokens
        + pricing["input_price_per_token"] * input_token_count
        + pricing["output_price_per_token"] * output_token_count
    ) / 1000000 + num_web_searches * web_search_cost_per_request

    result["cost_without_caching"] = (
        pricing["input_price_per_token"] * (cache_read_tokens + input_token_count)
        + pricing["output_price_per_token"] * output_token_count
    ) / 1000000 + num_web_searches * web_search_cost_per_request

    return result


def normalize_usage_dynamic(usage: dict, config: Dict[str, Any] = None) -> dict:
    """Main function with dynamic pricing"""
    if config is None:
        config = load_pricing_config()
        print(config)

    model_name = usage["model"]

    if "gemini" in model_name:
        result = normalize_usage_gemini_dynamic(usage, config["gemini"])
    elif "gpt" in model_name or "openai" in model_name:
        result = normalize_usage_gpt_dynamic(usage, config["openai"])
    elif "claude" in model_name or "anthropic" in model_name:
        result = normalize_usage_claude_dynamic(usage, config["anthropic"])
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Add common fields
    result["model"] = model_name
    system = os.path.basename(usage["file_path"]).replace(".json", "").split("-")
    if system[-1].startswith("0"):
        system = system[:-1]
    system = "-".join(system)
    if system.endswith("-expert"):
        system = "coe-system"
    result["system"] = system
    result["problem_id"] = usage["problem_id"]
    result["domain"] = usage["domain"]

    return result


# Modified audit_costs function
def audit_costs_dynamic(
    output_dir: str, cost_audit_csv_path: str, pricing_config_path: str = "eval/pricing_config.json"
):
    """Audit costs with dynamic pricing"""
    config = load_pricing_config(pricing_config_path)

    result = []
    for problem_id in os.listdir(output_dir):
        audit = cost_audit_dynamic(f"{output_dir}/{problem_id}", config)
        result.extend(audit)

    if not result:
        rprint("[bold red]No cost data found[/bold red]")
        return {"total_per_system": [], "grand_total": {"cost": 0, "cost_without_caching": 0}, "num_problems": 0}

    df = pd.DataFrame(result)
    df = df[["problem_id", "system", "model", "cost", "cost_without_caching"]]

    # Group by problem_id, system, model
    df = df.groupby(["problem_id", "system", "model"]).agg({"cost": "sum", "cost_without_caching": "sum"}).reset_index()

    num_problems = len(df["problem_id"].unique())
    rprint(f"[bold green]Detailed cost audit for {num_problems} problems:[/bold green]")
    rprint(df)
    df.to_csv(cost_audit_csv_path, index=False)

    # Average cost per system
    rprint(f"[bold yellow]Average cost per problem (over {num_problems} problems):[/bold yellow]")
    df_avg = df.groupby(["system", "problem_id"]).agg({"cost": "sum", "cost_without_caching": "sum"}).reset_index()
    df_avg = (
        df_avg.groupby(["system"]).agg({"cost": ["mean", "std"], "cost_without_caching": ["mean", "std"]}).reset_index()
    )

    # Flatten column names
    df_avg.columns = [
        "system",
        "cost_mean",
        "cost_std",
        "cost_without_caching_mean",
        "cost_without_caching_std",
    ]
    df_avg["cost"] = df_avg.apply(lambda x: f"{x['cost_mean']:.3f} ± {x['cost_std']:.3f}", axis=1)
    df_avg["cost_without_caching"] = df_avg.apply(
        lambda x: f"{x['cost_without_caching_mean']:.3f} ± {x['cost_without_caching_std']:.3f}",
        axis=1,
    )
    df_avg = df_avg[["system", "cost", "cost_without_caching"]]
    rprint(df_avg)
    rprint("\n")

    # Total cost per system
    rprint(f"[bold red]Total cost per system (across all {num_problems} problems):[/bold red]")
    total_costs = df.groupby(["system"]).agg({"cost": "sum", "cost_without_caching": "sum"}).reset_index()
    total_costs_display = total_costs.copy()
    total_costs_display["cost"] = total_costs_display["cost"].apply(lambda x: f"${x:.2f}")
    total_costs_display["cost_without_caching"] = total_costs_display["cost_without_caching"].apply(
        lambda x: f"${x:.2f}"
    )
    rprint(total_costs_display)

    # Total cost across all systems
    grand_total = df.agg({"cost": "sum", "cost_without_caching": "sum"})
    rprint(
        "[bold magenta]Total cost across all systems:"
        f" ${grand_total['cost']:.2f} (${grand_total['cost_without_caching']:.2f} without caching)[/bold magenta]"
    )
    rprint("\n")

    # Return cost summary data
    cost_summary = {
        "total_per_system": total_costs.to_dict("records"),
        "grand_total": {
            "cost": float(grand_total["cost"]),
            "cost_without_caching": float(grand_total["cost_without_caching"]),
        },
        "num_problems": num_problems,
    }

    return cost_summary


def compute_running_cost_and_estimate(
    system_results: List[AnnotatedSystemResult], total_problems: int, config: Dict[str, Any] = None
) -> Tuple[float, float]:
    """
    Compute running cost and estimate remaining cost

    Args:
        system_results: List of completed problems by the system
        total_problems: Total number of problems to solve
        config: Pricing configuration

    Returns:
        Tuple of (running_cost, estimated_remaining_cost)
    """
    if config is None:
        config = load_pricing_config()

    # Compute total cost for completed problems
    total_cost = 0.0
    for system_result in system_results:
        problem_cost = sum(x["cost"] for x in cost_audit_from_system_result(system_result, config))
        total_cost += problem_cost

    # Estimate remaining cost
    if len(system_results) > 0:
        avg_cost_per_problem = total_cost / len(system_results)
        remaining_problems = total_problems - len(system_results)
        estimated_remaining_cost = avg_cost_per_problem * remaining_problems
    else:
        estimated_remaining_cost = 0.0

    return total_cost, estimated_remaining_cost


def display_cost_progress(
    completed_problems: int, total_problems: int, running_cost: float, estimated_remaining: float
):
    """Display cost progress information"""
    progress_pct = (completed_problems / total_problems) * 100 if total_problems > 0 else 0
    estimated_total = running_cost + estimated_remaining

    rprint(f"[bold blue]Progress: {completed_problems}/{total_problems} ({progress_pct:.1f}%)[/bold blue]")
    rprint(f"[bold green]Running Cost: ${running_cost:.3f}[/bold green]")
    rprint(f"[bold yellow]Estimated Remaining: ${estimated_remaining:.3f}[/bold yellow]")
    rprint(f"[bold magenta]Estimated Total: ${estimated_total:.3f}[/bold magenta]")
    if completed_problems > 0:
        avg_cost = running_cost / completed_problems
        rprint(f"[bold cyan]Average Cost per Problem: ${avg_cost:.4f}[/bold cyan]")
    rprint("-" * 50)


def cost_audit_dynamic(problem_id_path: str, config: Dict[str, Any]) -> list[dict]:
    """Modified cost_audit to use dynamic pricing"""
    data = []
    for file in glob(f"{problem_id_path}/**/*.json", recursive=True):
        if "experts.json" in file:
            continue
        data.extend(cost_audit_from_file_dynamic(file, config))
    return data


def cost_audit_from_file_dynamic(file_path: str, config: Dict[str, Any]) -> list[dict]:
    """Modified cost_audit_from_file to use dynamic pricing"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return cost_audit_from_system_result(data, config)


def cost_audit_from_system_result(system_result: AnnotatedSystemResult, config: Dict[str, Any]) -> list[dict]:
    """Modified cost_audit_from_file to use dynamic pricing"""
    try:
        problem = Problem.model_validate(system_result["problem"])
    except Exception as e:
        # adding support for legacy where problem is a string
        if isinstance(system_result["problem"], str):
            problem_data = json.loads(system_result["problem"])
            problem = Problem.model_validate(problem_data)
        else:
            raise e
    usage_data = system_result["usage"]
    main_model = system_result["agent_config"]["model"]
    duration = system_result["duration"]

    if isinstance(usage_data, dict):
        usage_data = [usage_data]

    result = []
    for usage in usage_data:
        if "model" not in usage:
            usage["model"] = main_model
        usage["file_path"] = system_result["run_id"]
        usage["problem_id"] = problem.problem_id
        usage["domain"] = problem.domain

        # Clean up usage data
        if "completion_tokens_details" in usage:
            del usage["completion_tokens_details"]
        if "gemini" in usage["model"] and "prompt_tokens_details" in usage:
            del usage["prompt_tokens_details"]
        if "server_tool_use" in usage:
            tmp = usage["server_tool_use"]
            del usage["server_tool_use"]
            for k, v in tmp.items():
                usage[f"{k}"] = v

        usage["duration"] = duration
        result.append(normalize_usage_dynamic(usage, config))

    return result
