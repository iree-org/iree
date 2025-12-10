#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sets up a Python venv with compiler/runtime from a workflow run.

There are several modes in which to use this script:

* Locally, `--fetch-gh-workflow=WORKFLOW_ID` can be used to download and
  setup the venv from a specific workflow run in one step:


  ```bash
  python3.11 ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-gh-workflow=11977414405
  ```

* Locally, `--fetch-git-ref=GIT_REF` can be used to download and setup the
  venv from the latest workflow run for a given ref (commit) in one step:

  ```bash
  python3.11 ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-git-ref=main
  ```

* Locally, `--fetch-latest-main` can be used to download and setup the
  venv from the latest completed run from the `main` branch in one step:

  ```bash
  python3.11 ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-latest-main
  ```

* Within a workflow triggered by `workflow_call`, an artifact action will
  typically be used to fetch relevant package artifacts. Specify the fetched
  location with `--artifact-path=`:

  ```yml
  - uses: actions/download-artifact@fa0a91b85d4f404e444e00e005971372dc801d16 # v4.1.8
    with:
      name: linux_x86_64_release_packages
      path: ${{ env.PACKAGE_DOWNLOAD_DIR }}
  - name: Setup venv
    run: |
      ./build_tools/pkgci/setup_venv.py ${VENV_DIR} \
      --artifact-path=${PACKAGE_DOWNLOAD_DIR}
  ```

* Within a workflow triggered by `workflow_dispatch`, pass `artifact_run_id` as
  an input that developers must specify when running the workflow:

  ```yml
  on:
    workflow_dispatch:
      inputs:
      artifact_run_id:
        type: string
        default: ""

  ...
    steps:
    - name: Setup venv
      run: |
        ./build_tools/pkgci/setup_venv.py ${VENV_DIR} \
        --fetch-gh-workflow=${{ inputs.artifact_run_id }}
  ```

  (Note that these two modes are often combined to allow for workflow testing)

You must have a GitHub token with `repo` scope available as the `GH_TOKEN`
environment variable if you will be fetching artifacts.
"""

from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import argparse
import platform
import subprocess
import sys
import tempfile
import zipfile

from gh_utils import (
    fetch_gh_artifact,
    get_latest_workflow_run_id_for_main,
    get_latest_workflow_run_id_for_ref,
    list_gh_artifacts,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Setup venv")
    parser.add_argument(
        "venv_dir", type=Path, help="Directory in which to create the venv"
    )
    parser.add_argument("--artifact-path", help="Path in which to find/fetch artifacts")

    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument(
        "--fetch-gh-workflow",
        help="Fetch artifacts from a specific GitHub workflow using its run ID, like `12125722686`",
    )
    fetch_group.add_argument(
        "--fetch-latest-main",
        help="Fetch artifacts from the latest workflow run on the `main` branch",
        action="store_true",
    )
    fetch_group.add_argument(
        "--fetch-git-ref",
        help="Fetch artifacts for a specific git ref. Refs can be branch names (e.g. `main`), commit hashes (short like `abc123` or long), or tags (e.g. `iree-3.0.0`)",
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
    args = parser.parse_args(argv)
    return args


def find_venv_python(venv_path: Path) -> Optional[Path]:
    paths = [venv_path / "bin" / "python", venv_path / "Scripts" / "python.exe"]
    for p in paths:
        if p.exists():
            return p
    return None


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
        print(
            f"Package {package_name} not found in cache. Fetching from {artifact_name}..."
        )
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


def main(args):
    # Look up the workflow run for latest main.
    if args.fetch_latest_main:
        latest_gh_workflow = get_latest_workflow_run_id_for_main()
        args.fetch_gh_workflow = str(latest_gh_workflow)
        args.fetch_latest_main = ""
        return main(args)

    # Look up the workflow run for a ref.
    if args.fetch_git_ref:
        latest_gh_workflow = get_latest_workflow_run_id_for_ref(args.fetch_git_ref)
        args.fetch_git_ref = ""
        args.fetch_gh_workflow = str(latest_gh_workflow)
        return main(args)

    # Make sure we have an artifact path if fetching.
    if not args.artifact_path and args.fetch_gh_workflow:
        with tempfile.TemporaryDirectory() as td:
            args.artifact_path = td
            return main(args)

    artifact_prefix = f"{platform.system().lower()}_{platform.machine()}"
    wheels = []
    for package_stem, variant in [
        ("iree-base-compiler", args.compiler_variant),
        ("iree-base-runtime", args.runtime_variant),
    ]:
        wheels.append(
            find_wheel_for_variants(args, artifact_prefix, package_stem, variant)
        )
    print("\nInstalling wheels:", wheels)

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

    print(f"\nvenv setup complete at '{venv_path}'. Activate it with")
    print(f"  source {venv_path}/bin/activate")

    return 0


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))
