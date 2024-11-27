#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a Python venv for compiler/runtime from a commit.

This finds the latest pkgci workflow run for a given commit and then runs
setup_venv.py.

Prerequisites:
    Install gh (https://cli.github.com/) following instructions at
    https://github.com/cli/cli#installation and authenticate:

    ```bash
    gh auth login
    ```

    Python 3.11 (matching what PkgCI builds):

    ```bash
    # Using venv:
    sudo apt install python3.11 python3.11-dev python3.11-venv

    # Using pyenv (https://github.com/pyenv/pyenv):
    pyenv shell 3.11
    ```

Example usage:
    setup_venv_for_ref.py iree-3.1.0rc20241122
    setup_venv_for_ref.py iree-3.1.0rc20241122 --python-interpreter=python3.11

    setup_venv_for_ref.py 5b0740c97a33edce29e753b14b9ff04789afcc53
    source ~/.iree/bisect/5b0740c97a33edce29e753b14b9ff04789afcc53/.venv/bin/activate

For script maintenance, refer to the GitHub API docs:
  * https://docs.github.com/en/rest/actions/workflow-runs
"""

import argparse
import json
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()

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
    parser.add_argument(
        "--python-interpreter",
        help="The path to the Python interpreter to use. Must be compatible with packages (typically version 3.11)",
        default="python",
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


def main(args):
    print("------------------------------------------------------------------")
    print(f"Installing packages for ref: {args.ref}")
    print(f"  Using base working directory : '{args.work_dir}'")
    Path.mkdir(args.work_dir, parents=True, exist_ok=True)

    print("")
    latest_run_id = get_latest_workflow_run_id_for_ref(args.ref)
    artifacts_dir = args.work_dir / str(args.ref)
    Path.mkdir(artifacts_dir, exist_ok=True)

    venv_dir = artifacts_dir / ".venv"
    subprocess.check_call(
        [
            args.python_interpreter,
            str(THIS_DIR / "setup_venv.py"),
            str(venv_dir),
            "--artifact-path",
            str(artifacts_dir),
            "--fetch-gh-workflow",
            str(latest_run_id),
        ]
    )

    # Log which packages are installed.
    venv_python_interpreter = str(venv_dir / "bin" / "python")
    print("")
    print(f"Checking packages with 'pip freeze':")
    subprocess.check_call([venv_python_interpreter, "-m", "pip", "freeze"])

    print("------------------------------------------------------------------")
    print("")


if __name__ == "__main__":
    main(parse_arguments())
