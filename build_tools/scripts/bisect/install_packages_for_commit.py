#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""build_tools/scripts/bisect/install_packages_for_commit.py

This downloads Python packages from the pkgci_build_packages.yml step in the
pkgci.yml workflow then installs them. For script maintenance, refer to the
GitHub API docs:
  * https://docs.github.com/en/rest/actions/workflow-runs
  * https://docs.github.com/en/rest/actions/artifacts

These packages are installed:
  * iree-base-compiler
  * iree-base-runtime

Prerequisites:
    Install gh (https://cli.github.com/) following instructions at
    https://github.com/cli/cli#installation and authenticate:

    ```bash
    gh auth login
    ```

    Have a venv (https://docs.python.org/3/library/venv.html) active:

    ```bash
    python -m venv .venv
    source ./.venv/bin/activate
    ```

Example usage:
    install_packages_for_commit.py iree-3.1.0rc20241122
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

OWNER = "iree-org"
REPO = "iree"
WORKFLOW = "pkgci.yml"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Install packages from a pkgci workflow run at a specific commit"
    )
    parser.add_argument(
        "ref",
        help="The git ref (commit hash, branch name, tag name, etc.) to install packages from",
    )
    parser.add_argument(
        "--work-dir",
        help="The working directory to use. Defaults to ~/.iree/bisect/",
        default=Path.home() / ".iree" / "bisect",
        type=Path,
    )
    return parser.parse_args()


def get_latest_workflow_run_id_for_ref(ref: str) -> int:
    # Find the latest run of the pkgci.yml workflow for the given ref.
    workflow_run_args = [
        "gh",
        "api",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        f"/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW}/runs?head_sha={ref}",
    ]
    print(f"Running command to list workflow runs:\n  {' '.join(workflow_run_args)}")
    workflow_run_output = subprocess.check_output(workflow_run_args)
    workflow_run_json_output = json.loads(workflow_run_output)
    if workflow_run_json_output["total_count"] == 0:
        raise RuntimeError("Workflow did not run at this commit")

    latest_run = workflow_run_json_output["workflow_runs"][-1]
    print(f"Found workflow run: {latest_run['html_url']}")
    return latest_run["id"]


def download_artifacts_for_run_id(run_id: int, dir: Path):
    download_artifacts_args = [
        "gh",
        "run",
        "download",
        "--repo",
        f"{OWNER}/{REPO}",
        str(run_id),
        "--name",
        "linux_x86_64_release_packages",
        "--dir",
        str(dir),
    ]
    print(
        f"Running command to download artifacts:\n  {' '.join(download_artifacts_args)}"
    )
    subprocess.check_call(download_artifacts_args)


def install_packages_from_directory(dir: Path):
    # Setup venv.
    venv_dir = dir / ".venv"
    print(f"Creating venv at '{venv_dir}'")
    subprocess.check_call(["python", "-m", "venv", str(venv_dir)])
    venv_python_interpreter = str(venv_dir / "bin" / "python")
    subprocess.check_call(
        [venv_python_interpreter, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
    )

    # Install common deps.
    install_deps_args = [
        venv_python_interpreter,
        "-m",
        "pip",
        "install",
        "--quiet",
        "numpy",
        "sympy",
    ]
    print("")
    print(f"Running command to install dependencies:\n  {' '.join(install_deps_args)}")
    subprocess.check_call(install_deps_args)

    # Install each .whl in the directory.
    whl_files = list(dir.glob("*.whl"))
    for file in whl_files:
        install_package_args = [
            venv_python_interpreter,
            "-m",
            "pip",
            "install",
            "--quiet",
            str(file),
        ]
        print(
            f"Running command to install package:\n  {' '.join(install_package_args)}"
        )
        subprocess.check_call(install_package_args)

    # Log which packages are installed.
    print("")
    print(f"Checking packages with 'pip freeze':")
    subprocess.check_call([venv_python_interpreter, "-m", "pip", "freeze"])


def main(args):
    print("------------------------------------------------------------------")
    print(f"Installing packages for ref: {args.ref}")
    print(f"  Using base working directory : '{args.work_dir}'")
    Path.mkdir(args.work_dir, parents=True, exist_ok=True)

    print("")
    latest_run_id = get_latest_workflow_run_id_for_ref(args.ref)
    artifacts_dir = args.work_dir / str(args.ref)
    Path.mkdir(artifacts_dir, exist_ok=True)

    existing_files = list(artifacts_dir.glob("*.whl"))
    if existing_files:
        print("Found cached .whl files in artifacts dir, skipping download")
    else:
        download_artifacts_for_run_id(latest_run_id, artifacts_dir)

    install_packages_from_directory(artifacts_dir)

    print("------------------------------------------------------------------")
    print("")


if __name__ == "__main__":
    main(parse_arguments())
