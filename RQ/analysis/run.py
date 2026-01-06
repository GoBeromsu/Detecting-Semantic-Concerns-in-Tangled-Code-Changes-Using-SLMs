#!/usr/bin/env python3
"""
Unified RQ Analysis Runner
Single entry point for running all research question analyses.

Usage:
    python run.py --rq 1          # Run RQ1 only
    python run.py --rq 1 2 3      # Run RQ1, RQ2, RQ3
    python run.py --all           # Run all RQs
    python run.py --list          # List available scripts
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from typing import List, Optional


def load_config() -> dict:
    """Load unified configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_scripts(config: dict) -> None:
    """List all available scripts by RQ."""
    print("\nAvailable Research Questions and Scripts:")
    print("=" * 60)

    for rq_num in range(1, 5):
        rq_key = f"rq{rq_num}"
        if rq_key in config:
            rq_config = config[rq_key]
            print(f"\nRQ{rq_num}: {rq_config['name']}")
            print(f"  {rq_config['description']}")
            print("  Scripts:")
            # scripts is a dict: {script_name: {description, file, ...}}
            for script_name, script_config in rq_config["scripts"].items():
                description = script_config.get("description", "No description")
                print(f"    - {script_name}: {description}")


def run_rq(rq_num: int, config: dict, base_dir: Path, project_root: Path) -> bool:
    """Run all scripts for a specific RQ using python -m for proper imports."""
    rq_key = f"rq{rq_num}"

    if rq_key not in config:
        print(f"Error: RQ{rq_num} not found in configuration")
        return False

    rq_config = config[rq_key]
    rq_dir = base_dir / f"RQ{rq_num}"

    if not rq_dir.exists():
        print(f"Error: Directory {rq_dir} not found")
        return False

    print(f"\nRQ{rq_num}: {rq_config['name']}")
    print("=" * 60)

    all_success = True

    # scripts is a dict: {script_name: {description, file, ...}}
    for script_name, script_config in rq_config["scripts"].items():
        # Determine script file name
        # If 'file' is specified, use it; otherwise derive from script_name
        script_file = script_config.get("file", f"{script_name}.py")
        # Handle cases like "concerncount-by-model" -> "concerncount-by-model.py"
        if not script_file.endswith(".py"):
            script_file = f"{script_file}.py"

        script_path = rq_dir / script_file

        if not script_path.exists():
            print(f"  [SKIP] {script_name}: File not found ({script_file})")
            continue

        description = script_config.get("description", "No description")
        print(f"\n  Running {script_name}...")
        print(f"  Description: {description}")

        # Convert file name to module name (e.g., "concern_count_pairwise_pvalue.py" -> "concern_count_pairwise_pvalue")
        module_name = script_file.replace(".py", "").replace("-", "_")
        full_module = f"RQ.analysis.RQ{rq_num}.{module_name}"

        # Build command with optional csv_files arguments
        cmd = [sys.executable, "-m", full_module]

        # If script has csv_files in config, pass them as arguments
        if "csv_files" in script_config:
            for csv_file in script_config["csv_files"]:
                cmd.append(str(project_root / csv_file))

        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=False,
            )

            if result.returncode == 0:
                print(f"  [SUCCESS] {script_name} completed")
            else:
                print(f"  [FAILED] {script_name} (exit code: {result.returncode})")
                all_success = False

        except Exception as e:
            print(f"  [ERROR] {script_name}: {e}")
            all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Unified RQ Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --rq 1          # Run RQ1 only
  python run.py --rq 1 2 3      # Run RQ1, RQ2, RQ3
  python run.py --all           # Run all RQs (1-4)
  python run.py --list          # List available scripts
        """,
    )
    parser.add_argument(
        "--rq",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        help="Specific RQ number(s) to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all RQs (1-4)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scripts",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    base_dir = Path(__file__).parent
    project_root = base_dir.parent.parent  # RQ/analysis -> RQ -> project root

    # Handle --list
    if args.list:
        list_scripts(config)
        return

    # Determine which RQs to run
    rq_numbers: List[int] = []

    if args.all:
        rq_numbers = [1, 2, 3, 4]
    elif args.rq:
        rq_numbers = args.rq
    else:
        parser.print_help()
        print("\nError: Please specify --rq, --all, or --list")
        sys.exit(1)

    # Run selected RQs
    print(f"\nRunning RQ(s): {', '.join(map(str, rq_numbers))}")

    results = {}
    for rq_num in rq_numbers:
        results[rq_num] = run_rq(rq_num, config, base_dir, project_root)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for rq_num, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  RQ{rq_num}: {status}")

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
