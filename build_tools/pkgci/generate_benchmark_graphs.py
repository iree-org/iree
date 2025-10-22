#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import sys
import os


def process_summary(commit_hash: str, summary: dict) -> dict:
    # Process the summary to extract relevant information.
    processed = {"commit_hash": commit_hash, "tests": []}
    for test in summary["benchmark"]["rows"]:
        processed["tests"].append(
            {"name": test[0].removesuffix(".json").replace("/", "_"), "time": test[1]}
        )
    return processed


def append_history(commit_hash: str, results_json_path: str, results_history_path: str):
    # Get the results for the current run.
    with open(results_json_path, "r") as f:
        results = process_summary(commit_hash, json.load(f))

    # Append the results to the history.
    results_history = []
    max_history = 100
    if os.path.exists(results_history_path):
        with open(results_history_path, "r") as f:
            results_history = json.load(f)
    results_history.append(results)
    # Keep only the most recent results.
    if len(results_history) > max_history:
        results_history = results_history[-max_history:]

    # Write the updated history back to the file.
    os.makedirs(os.path.dirname(results_history_path), exist_ok=True)
    with open(results_history_path, "w") as f:
        json.dump(results_history, f, indent=2)


def generate_html(results_history: list):
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

    # Start building the HTML content
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


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python performance_publish.py <commint_hash> <path_to_results_json> <path_to_results_history> <path_to_results_html>\n"
            "This script reads the performance results from the specified JSON file, appends them to the history file, and generates an HTML visualization.\n"
        )
        sys.exit(1)

    commit_hash = sys.argv[1]
    results_json_path = sys.argv[2]
    results_history_path = sys.argv[3]
    results_html_path = sys.argv[4]
    append_history(commit_hash, results_json_path, results_history_path)

    # Generate and save the HTML file
    results_history = []
    with open(results_history_path, "r") as f:
        results_history = json.load(f)
    html_content = generate_html(results_history)
    with open(results_html_path, "w") as f:
        f.write(html_content)
