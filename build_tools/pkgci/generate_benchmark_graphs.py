#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import argparse
import json
import tempfile
import zipfile

from pathlib import Path
from typing import List, Dict

from gh_utils import (
    list_gh_artifacts,
    fetch_gh_artifact,
    get_commit_from_run_id,
)

# Benchmark workflows to track. This needs to be consistent with
# summary-name fields defined in pkgci_test_torch.yml.
benchmark_workflows = ["torch_models_cpu_task", "torch_models_amdgpu_mi325"]

max_history_length = 200


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Compare CI Benchmarks")
    parser.add_argument(
        "--workflow-run-id",
        help="Fetch artifacts from a specific GitHub workflow using its run ID, like `12125722686`",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to output files",
    )
    args = parser.parse_args(argv)
    return args


def get_benchmark_artifacts_zip_links(run_id: int) -> Dict:
    artifacts = list_gh_artifacts(run_id)
    benchmark_artifacts = {}
    for workflow_name in benchmark_workflows:
        # Each benchmark workflow uploads a JSON job summary as a zip artifact.
        artifact_name = f"{workflow_name}_summary.json"
        if artifact_name in artifacts:
            benchmark_artifacts[workflow_name] = artifacts[artifact_name]
    return benchmark_artifacts


def extract_benchmark_json_from_zip(tmpdir_path: Path, url: str) -> Path:
    zip_path = tmpdir_path / "temp.zip"
    fetch_gh_artifact(url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(tmpdir_path)
    json_path = tmpdir_path / "job_summary.json"
    return json_path


def process_data(commit_hash: str, data: Dict) -> Dict:
    # Process data to get relevant information.
    processed = {"commit_hash": commit_hash, "tests": []}
    for test in data["benchmark"]["rows"]:
        processed["tests"].append(
            {"name": test[0].removesuffix(".json").replace("/", "_"), "time": test[1]}
        )
    return processed


def append_data_to_history(
    commit_hash: str, new_result_json: Path, history_results_json: Path
):
    # Get the results for the current run.
    with open(new_result_json, "r") as f:
        new_results = json.load(f)
        new_results = process_data(commit_hash, new_results)

    # Append the results to the history.
    if history_results_json.exists():
        with open(history_results_json, "r") as f:
            history_results = json.load(f)
    else:
        history_results = []
    history_results.append(new_results)

    # Keep only the most recent results.
    if len(history_results) > max_history_length:
        history_results = history_results[-max_history_length:]

    # Write the updated history back to the file.
    history_results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(history_results_json, "w") as f:
        json.dump(history_results, f, indent=2)


def dump_html(results_history: List) -> str:
    graph_data = {}
    for entry in results_history:
        for test in entry["tests"]:
            name = test["name"]
            if name not in graph_data:
                graph_data[name] = {
                    "commit_hashes": [],
                    "time": [],
                }

    time_unit = "ms"
    for entry in results_history:
        commit_hash = str(entry["commit_hash"])[:7]
        local_tests = dict.fromkeys(graph_data.keys(), None)
        for test in entry["tests"]:
            local_tests[test["name"]] = test
        for test_name, test in local_tests.items():
            graph_data[test_name]["commit_hashes"].append(commit_hash)
            # So that the time/commit horizontal/x axis is consistent
            # across tests, even if a test is missing for a commit,
            # we add time=0 if a test did not run successfully.
            if not test:
                graph_data[test_name]["time"].append(0)
            else:
                graph_data[test_name]["time"].append(test["time"])

    # Start building the HTML content.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Tracker</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            .chart-container { width: 80%; margin: 30px auto; }
            canvas { width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Benchmark Tracker</h1>
    """

    # Add a graph for each test
    for test_name, data in graph_data.items():
        html_content += f"""
        <div class="chart-container">
            <h2>{test_name}</h2>"""
        html_content += f"""
            <canvas id="chart-{test_name.replace(' ', '-')}"></canvas>
        </div>
        <script>
            const ctx_{test_name.replace(' ', '_')} = document.getElementById('chart-{test_name.replace(' ', '-')}')
            const chart_{test_name.replace(' ', '_')} = new Chart(ctx_{test_name.replace(' ', '_')}, {{
                type: 'line',
                data: {{
                    labels: {data["commit_hashes"]},  // Truncated commit hashes as X-axis labels
                    datasets: [{{
                        label: 'Time ({time_unit})',
                        data: {data["time"]},   // Test time as Y-axis
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Commit Hash'
                            }}
                        }},
                        y: {{
                            beginAtZero: true,  // Ensures the Y-axis starts at 0
                            title: {{
                                display: true,
                                text: 'Time ({time_unit})'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """

    # Close the HTML content
    html_content += """
    </body>
    </html>
    """

    return html_content


def process_and_generate(
    commit_hash: str,
    new_results_json: Path,
    history_results_json: Path,
    html_path: Path,
):
    # Append new data to history.
    append_data_to_history(commit_hash, new_results_json, history_results_json)
    # Generate and save the HTML file.
    with open(history_results_json, "r") as f:
        results_history = json.load(f)
    html_content = dump_html(results_history)
    with open(html_path, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    commit_hash = get_commit_from_run_id(args.workflow_run_id)
    artifact_links = get_benchmark_artifacts_zip_links(args.workflow_run_id)

    # Process each benchmark workflow.
    for workflow_name, artifact_url in artifact_links.items():
        history_results_json = output_dir / f"{workflow_name}_history.json"
        html_path = output_dir / f"{workflow_name}.html"
        with tempfile.TemporaryDirectory() as tmpdir:
            new_results_json = extract_benchmark_json_from_zip(
                Path(tmpdir), artifact_url
            )
            process_and_generate(
                commit_hash, new_results_json, history_results_json, html_path
            )
