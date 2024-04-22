#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a Python venv with compiler/runtime from a workflow run.

There are two modes in which to use this script:

* Within a workflow, an artifact action will typically be used to fetch
  relevant package artifacts. Specify the fetch location with
  `--artifact-path=`.

* Locally, the `--fetch-gh-workflow=WORKFLOW_ID` can be used instead in order
  to download and setup the venv in one step.

You must have the `gh` command line tool installed and authenticated if you
will be fetching artifacts.
"""

from typing import Optional, Dict, Tuple

import argparse
import functools
from glob import glob
import json
import os
import sys
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


@functools.lru_cache
def list_gh_artifacts(run_id: str) -> Dict[str, str]:
    print(f"Fetching artifacts for workflow run {run_id}")
    base_path = f"/repos/iree-org/iree"
    output = subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            f"{base_path}/actions/runs/{run_id}/artifacts",
        ]
    )
    data = json.loads(output)
    # Uncomment to debug:
    # print(json.dumps(data, indent=2))
    artifacts = {
        rec["name"]: f"{base_path}/actions/artifacts/{rec['id']}/zip"
        for rec in data["artifacts"]
    }
    print("Found artifacts:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")
    return artifacts


def fetch_gh_artifact(api_path: str, file: Path):
    print(f"Downloading artifact {api_path}")
    contents = subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            api_path,
        ]
    )
    file.write_bytes(contents)


def find_venv_python(venv_path: Path) -> Optional[Path]:
    paths = [venv_path / "bin" / "python", venv_path / "Scripts" / "python.exe"]
    for p in paths:
        if p.exists():
            return p
    return None


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Setup venv")
    parser.add_argument("--artifact-path", help="Path in which to find/fetch artifacts")
    parser.add_argument(
        "--fetch-gh-workflow", help="Fetch artifacts from a GitHub workflow"
    )
    parser.add_argument(
        "--compiler-variant",
        default="",
        help="Package variant to install for the compiler ('', 'asserts')",
    )
    parser.add_argument(
        "--runtime-variant",
        default="",
        help="Package variant to install for the runtime ('', 'asserts')",
    )
    parser.add_argument(
        "venv_dir", type=Path, help="Directory in which to create the venv"
    )
    args = parser.parse_args(argv)
    return args


def main(args):
    # Make sure we have an artifact path if fetching.
    if not args.artifact_path and args.fetch_gh_workflow:
        with tempfile.TemporaryDirectory() as td:
            args.artifact_path = td
            return main(args)

    # Find the regression suite project.
    rs_dir = (
        (Path(__file__).resolve().parent.parent.parent)
        / "experimental"
        / "regression_suite"
    )
    if not rs_dir.exists():
        print(f"Could not find regression_suite project: {rs_dir}")
        return 1

    artifact_prefix = f"{platform.system().lower()}_{platform.machine()}"
    wheels = []
    for package_stem, variant in [
        ("iree-compiler", args.compiler_variant),
        ("iree-runtime", args.runtime_variant),
    ]:
        wheels.append(
            find_wheel_for_variants(args, artifact_prefix, package_stem, variant)
        )
    print("Installing wheels:", wheels)

    # Set up venv.
    venv_path = args.venv_dir
    python_exe = find_venv_python(venv_path)

    if not python_exe:
        print(f"Creating venv at {str(venv_path)}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        python_exe = find_venv_python(venv_path)
        if not python_exe:
            raise RuntimeError("Error creating venv")

    # Install each of the built wheels without deps or consulting an index.
    # This is because we absolutely don't want this falling back to anything
    # but what we said.
    for artifact_path, package_name in wheels:
        cmd = [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "-f",
            str(artifact_path),
            "--force-reinstall",
            package_name,
        ]
        print(f"Running command: {' '.join([str(c) for c in cmd])}")
        subprocess.check_call(cmd)

    # Now install the regression suite project, which will bring in any deps.
    cmd = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--timeout",
        "60",
        "-e",
        str(rs_dir) + os.sep,
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    return 0


def find_wheel_for_variants(
    args, artifact_prefix: str, package_stem: str, variant: str
) -> Tuple[Path, str]:
    artifact_path = Path(args.artifact_path)
    package_suffix = "" if variant == "" else f"-{variant}"
    package_name = f"{package_stem}{package_suffix}"

    def has_package():
        norm_package_name = package_name.replace("-", "_")
        pattern = str(artifact_path / f"{norm_package_name}-*.whl")
        files = glob(pattern)
        return bool(files)

    if has_package():
        return (artifact_path, package_name)

    if not args.fetch_gh_workflow:
        raise RuntimeError(
            f"Could not find package {package_name} to install from {artifact_path}"
        )

    # Fetch.
    artifact_path.mkdir(parents=True, exist_ok=True)
    artifact_suffix = "" if variant == "" else f"_{variant}"
    artifact_name = f"{artifact_prefix}_release{artifact_suffix}_packages"
    artifact_file = artifact_path / f"{artifact_name}.zip"
    if not artifact_file.exists():
        print(f"Package {package_name} not found. Fetching from {artifact_name}...")
        artifacts = list_gh_artifacts(args.fetch_gh_workflow)
        if artifact_name not in artifacts:
            raise RuntimeError(
                f"Could not find required artifact {artifact_name} in run {args.fetch_gh_workflow}"
            )
        fetch_gh_artifact(artifacts[artifact_name], artifact_file)
    print(f"Extracting {artifact_file}")
    with zipfile.ZipFile(artifact_file) as zip_ref:
        zip_ref.extractall(artifact_path)

    # Try again.
    if not has_package():
        raise RuntimeError(f"Could not find {package_name} in {artifact_path}")
    return (artifact_path, package_name)


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
