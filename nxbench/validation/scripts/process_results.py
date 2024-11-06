import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_results_file(file_path: Path):
    """
    Process a single asv results JSON file to extract only execution_time.
    """
    try:
        with file_path.open("r") as f:
            data = json.load(f)

        results = data.get("results", {})
        modified = False

        for benchmark_name, benchmark_data in results.items():
            if not isinstance(benchmark_data, list) or len(benchmark_data) < 1:
                logger.warning(
                    f"Unexpected format for benchmark '{benchmark_name}' in file '{file_path}'"
                )
                continue

            runs = benchmark_data[0]
            if not isinstance(runs, list):
                logger.warning(
                    f"Unexpected runs format for benchmark '{benchmark_name}' in file '{file_path}'"
                )
                continue

            new_runs = []
            for run in runs:
                if isinstance(run, dict):
                    execution_time = run.get("execution_time", None)
                    if execution_time is not None:
                        new_runs.append(execution_time)
                        modified = True
                    else:
                        logger.warning(
                            f"No 'execution_time' found in run for benchmark '{benchmark_name}' in file '{file_path}'"
                        )
                        new_runs.append(None)
                else:
                    logger.warning(
                        f"Unexpected run format for benchmark '{benchmark_name}' in file '{file_path}': {run}"
                    )
                    new_runs.append(None)

            data["results"][benchmark_name][0] = new_runs

        if modified:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            shutil.copy(file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")

            with file_path.open("w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Processed and updated file: {file_path}")
        else:
            logger.info(f"No modifications needed for file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process file '{file_path}': {e}")


def process_all_results(results_dir: Path):
    """
    Traverse the results directory and process all JSON files.
    """
    if not results_dir.exists() or not results_dir.is_dir():
        logger.error(
            f"Results directory '{results_dir}' does not exist or is not a directory."
        )
        return

    json_files = list(results_dir.rglob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in '{results_dir}'.")
        return

    logger.info(f"Found {len(json_files)} JSON files in '{results_dir}'.")

    for json_file in json_files:
        logger.info(f"Processing file: {json_file}")
        process_results_file(json_file)


def main():
    parser = argparse.ArgumentParser(
        description="Process asv results to extract execution_time."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the asv results directory (default: results)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    process_all_results(results_dir)


if __name__ == "__main__":
    main()
