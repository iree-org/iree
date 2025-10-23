#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
from typing import Dict
import json
import tempfile
import zipfile

from gh_utils import (
    get_latest_workflow_run_id_for_main,
    get_latest_workflow_run_id_for_ref,
    list_gh_artifacts,
    fetch_gh_artifact,
    get_commit_from_run_id,
)

benchmark_workflows = {
    "torch_models_cpu_task_summary.json": "cpu",
    "torch_models_amdgpu_mi325_summary.json": "mi325",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Compare CI Benchmarks")
    parser.add_argument(
        "--workflow-run-id",
        help="Fetch artifacts from a specific GitHub workflow using its run ID, like `12125722686`",
    )
    parser.add_argument(
        "--output-file-path",
        help="Path to output markdown file",
        default="./job_summary.md",
    )
    args = parser.parse_args(argv)
    return args


def get_benchmark_artifacts_zip_links(run_id: int):
    artifacts = list_gh_artifacts(run_id)
    benchmark_artifacts = {}
    for benchmark, _ in benchmark_workflows.items():
        if benchmark in artifacts:
            benchmark_artifacts[benchmark] = artifacts[benchmark]
    return benchmark_artifacts


def get_benchmark_data_from_zip(url: str) -> Dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "temp.zip"
        fetch_gh_artifact(url, zip_path)
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(tmpdir_path)
        with open(tmpdir_path / "job_summary.json", "r") as f:
            data = json.load(f)
    return data


def get_benchmark_data_from_summary(summary: Dict):
    rows = summary["benchmark"]["rows"]
    outputs = {}
    for row in rows:
        # 'Name', 'Current Time (ms)', 'Golden Time (ms)', 'Status'
        name, current_time, _, _ = row
        outputs[name] = {
            "current_time": float(current_time),
        }
    return outputs


def main(args):
    run_id = get_latest_workflow_run_id_for_main()
    main_benchmarks = get_benchmark_artifacts_zip_links(run_id)
    pr_benchmarks = get_benchmark_artifacts_zip_links(args.workflow_run_id)
    # Take intersection of keys
    common_keys = set(main_benchmarks.keys()).intersection(set(pr_benchmarks.keys()))

    comparison_benchmarks: Dict[str, tuple[Dict, Dict]] = {}
    for key in common_keys:
        main_data = get_benchmark_data_from_zip(main_benchmarks[key])
        pr_data = get_benchmark_data_from_zip(pr_benchmarks[key])
        comparison_benchmarks[key] = (main_data, pr_data)

    benchmark_outputs = {}
    for key, (main_data, pr_data) in comparison_benchmarks.items():
        try:
            main_benchmarks = get_benchmark_data_from_summary(main_data)
            pr_benchmarks = get_benchmark_data_from_summary(pr_data)
            # Get a name, main_time, pr_time dict.
            benchmark_outputs[key] = {
                name: [
                    main_benchmarks[name]["current_time"],
                    pr_benchmarks[name]["current_time"],
                    (
                        (
                            pr_benchmarks[name]["current_time"]
                            - main_benchmarks[name]["current_time"]
                        )
                        / main_benchmarks[name]["current_time"]
                    )
                    * 100
                    if main_benchmarks[name]["current_time"] != 0
                    else "N/A",
                ]
                for name in pr_benchmarks.keys()
                if name in main_benchmarks
            }
            for name in pr_benchmarks.keys():
                if name not in main_benchmarks:
                    benchmark_outputs[key][name] = [
                        None,
                        pr_benchmarks[name]["current_time"],
                        "N/A",
                    ]
        except Exception as e:
            print(f"Error processing benchmark {key}: {e}")
            continue

    main_commit = get_commit_from_run_id(run_id)
    pr_commit = get_commit_from_run_id(args.workflow_run_id)

    with open(args.output_file_path, "w") as f:
        f.write("## Benchmark Comparison Report\n\n")
        f.write(f"HEAD {main_commit} vs PR {pr_commit}\n\n")
        for key, benchmarks in benchmark_outputs.items():
            dev = benchmark_workflows[key]
            f.write(f"### Benchmark Comparison for {dev}\n\n")
            f.write("| Benchmark | Main Time (ms) | PR Time (ms) | Change (%) |\n")
            f.write("|-----------|----------------|---------------|--------|\n")
            for name, (main_time, pr_time, change) in benchmarks.items():
                main_time_str = f"{main_time:.2f}" if main_time is not None else "N/A"
                pr_time_str = f"{pr_time:.2f}" if pr_time is not None else "N/A"
                change_str = f"{change:.2f}%" if isinstance(change, float) else change
                f.write(
                    f"| {name} | {main_time_str} | {pr_time_str} | {change_str} |\n"
                )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
