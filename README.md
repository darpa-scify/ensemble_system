# Ensembler

This repository contains an ensemble system for combining multiple model outputs and an evaluation harness for testing the system.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/darpa-scify/ensemble_system.git
cd ensembler
```

### 2. Install Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download Evaluation Data

Download the evaluation data from Google Drive:
https://drive.google.com/drive/folders/1hU7YP6HVUc7CGnSZy1E4boaTuLbn6nmS?usp=drive_link

### 4. Download and Extract Runs Data

Download the runs data from Google Drive link below and extract it to the `runs` folder:
https://drive.google.com/drive/folders/1VwJ6lPLuez-KQVQviFGZIJVqbe9MR5OI?usp=drive_link

```
# Create runs directory if it doesn't exist
mkdir -p runs

# Extract all downloaded zip files to the runs folder
# This will extract all zip files in the current directory to the runs folder
for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting $zip_file..."
        unzip "$zip_file" -d runs/
    fi
done

# Remove all zip files after extraction (optional)
rm *.zip
```

### 5. Anthropic access required
Add an environment file with API key for anthropic

## Using the eval harness to run the ensembler script

> [!NOTE]
> The shell commands here are written assuming that you are in the project root of the project that *contains*
> the eval harness as a submodule, not the `eval_harness` dir itself.

```shell
$ python eval_harness/run.py \
  --problem-file path/to/problems.jsonl \  # comma-sep or glob
  --output-dir path/to/outputs \  # will write results to output-dir/{problem-id}/{run-id}.json
  --system ensemble_system.ensemble:baseline_ensemble \  
  [--run-id human-readable] \  # defaults to the system name
  [--exclude problem_id_0001] \  # comma-sep or glob
  [--force-rerun] \
  [--run-eval --gold-file path/to/golds.jsonl] \  # comma-sep or glob
  [--show-running-costs]
```

### Usage Notes

If you pass the same `--output-dir` and `--run-id` for an already-completed (or partially-completed) run, the eval
harness will not rerun the system for previously completed problems. This can be used to rerun eval metrics with
different exclusion sets or to recalculate costs.

### Examples

**One file**

```shell
python eval_harness/run.py --system ensemble_system.ensemble:baseline_ensemble \
  --output-dir runs/eval-harness-test \
  --problem-file "evaluation-data/subsets/gold-T&E/sprint1-problems.jsonl" \
  --run-eval --gold-file "evaluation-data/subsets/gold-T&E/sprint1-gold-standards.jsonl" \
  --force-rerun --show-running-costs
```

**Multiple input files**

```shell
python eval_harness/run.py --system ensemble_system.ensemble:baseline_ensemble \
  --output-dir runs/eval-harness-test \
  --problem-file "evaluation-data/subsets/*/*problems*.jsonl" \
  --run-eval --gold-file "evaluation-data/subsets/*/*gold*.jsonl"
```

**Excluding problem IDs**

```shell
python eval_harness/run.py --system ensemble_system.ensemble:baseline_ensemble \
  --output-dir runs/eval-harness-test \
  --problem-file "evaluation-data/subsets/gold-T&E/sprint1-problems.jsonl" \
  --run-eval --gold-file "evaluation-data/subsets/gold-T&E/sprint1-gold-standards.jsonl" \
  --exclude "alloys_*" \
  --force-rerun --show-running-costs
```


### Multiple Input Files

`--problem-file` and `--gold-file` can take multiple inputs to test models on large sets of problems at once.

To pass multiple inputs, you may:
- Pass the option multiple times (e.g. `--problem-file a.jsonl --problem-file b.jsonl`)
- Separate filenames by commas (e.g. `--problem-file a.jsonl,b.jsonl`)
- Use a glob (e.g. `--problem-file "*.jsonl"`)
- Any combination of the above

### Excluding Problem IDs

`--exclude` can take multiple inputs to exclude certain problems from eval. You may pass multiple inputs in the same
manner as input files (including glob patterns).