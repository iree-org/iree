# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import json
import os
import urllib.request
from pathlib import Path
import subprocess
from typing import Dict

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent
BASE_API_PATH = "/repos/iree-org/iree"


def query_gh_api(api_path: str):
    url = f"https://api.github.com{api_path}"
    print(f"Querying GitHub API: {url}")
    gh_token = os.environ.get("GH_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {gh_token}",
    }
    req = urllib.request.Request(url)
    for k, v in headers.items():
        # https://github.com/arduino/report-size-deltas/pull/83
        req.add_unredirected_header(k, v)

    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise RuntimeError(
                f"GitHub API request failed: {response.status} {response.reason}"
            )
        contents = response.read()
        return contents


def get_latest_workflow_run_id_for_main() -> int:
    print(f"Looking up latest workflow run for main branch")
    # Note: at a high level, we probably want to select one of these:
    #   A) The latest run that produced package artifacts
    #   B) The latest run that passed all checks
    # Instead, we just check for the latest completed workflow. This can miss
    # runs that are still pending (especially if jobs are queued waiting for
    # runners) and can include jobs that failed tests (for better or worse).
    workflow_run_output = query_gh_api(
        f"{BASE_API_PATH}/actions/workflows/pkgci.yml/runs?branch=main&event=push&status=completed&per_page=1"
    )
    workflow_run_json_output = json.loads(workflow_run_output)
    latest_run = workflow_run_json_output["workflow_runs"][0]
    print(f"\nFound workflow run: {latest_run['html_url']}")
    return latest_run["id"]


def get_latest_workflow_run_id_for_ref(ref: str) -> int:
    print(f"Finding workflow run for ref: {ref}")
    normalized_ref = (
        subprocess.check_output(["git", "rev-parse", ref], cwd=REPO_ROOT)
        .decode()
        .strip()
    )

    print(f"  Using normalized ref: {normalized_ref}")
    workflow_run_output = query_gh_api(
        f"{BASE_API_PATH}/actions/workflows/pkgci.yml/runs?head_sha={normalized_ref}"
    )
    workflow_run_json_output = json.loads(workflow_run_output)
    if workflow_run_json_output["total_count"] == 0:
        raise RuntimeError("Workflow did not run at this commit")

    latest_run = workflow_run_json_output["workflow_runs"][-1]
    print(f"\nFound workflow run: {latest_run['html_url']}")
    return latest_run["id"]


@functools.lru_cache
def list_gh_artifacts(run_id: str) -> Dict[str, str]:
    print(f"Fetching artifacts for workflow run: {run_id}")
    output = query_gh_api(f"{BASE_API_PATH}/actions/runs/{run_id}/artifacts")
    data = json.loads(output)
    # Uncomment to debug:
    # print(json.dumps(data, indent=2))
    artifacts = {
        rec["name"]: f"{BASE_API_PATH}/actions/artifacts/{rec['id']}/zip"
        for rec in data["artifacts"]
    }
    print("\nFound artifacts:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")
    return artifacts


def get_commit_from_run_id(run_id: int) -> str:
    print(f"Fetching commit for workflow run: {run_id}")
    output = query_gh_api(f"{BASE_API_PATH}/actions/runs/{run_id}")
    data = json.loads(output)
    commit = data["head_sha"]
    print(f"Found commit: {commit}")
    return commit


def fetch_gh_artifact(api_path: str, file: Path):
    print(f"Downloading artifact {api_path}")
    contents = query_gh_api(api_path)
    file.write_bytes(contents)
