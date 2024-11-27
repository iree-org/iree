#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dev package bisect script.

This connects the `git bisect` tool (https://git-scm.com/docs/git-bisect)
with IREE's package builds, allowing developers to run tests through commit
history efficiently. For example, this can be used to spot at which commit
an `iree-compile` command started failing.

Requirements:
    git     (https://git-scm.com/)
    gh      (https://cli.github.com/)
    Linux   (at least until IREE builds packages for other systems at each commit)
    Python 3.11

Example usage:
    bisect_packages.py \
        --good-ref=iree-3.0.0 \
        --bad-ref=iree-3.1.0rc20241122 \
        --test-script=bisect_example_timestamp.sh
"""


import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent.parent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Git release bisect tool")
    # TODO(scotttodd): add --interactive mode that prompts like git bisect does
    parser.add_argument(
        "--good-ref",
        help="The git ref (commit hash, branch name, tag name, etc.) at the lower end of the range",
        required=True,
    )
    parser.add_argument(
        "--bad-ref",
        help="The git ref (commit hash, branch name, tag name, etc.) at the upper end of the range",
        required=True,
    )
    parser.add_argument(
        "--work-dir",
        help="The working directory to use. Defaults to ~/.iree/bisect/",
        default=Path.home() / ".iree" / "bisect",
        type=Path,
    )
    # TODO(scotttodd): choice between manual or script (`git bisect run`) to use
    #                  note that a "manual" mode would need developers to run
    #   1. `setup_venv_for_ref.py $(git rev-parse BISECT_HEAD)`
    #   2. `source $WORKDIR/$(git rev-parse BISECT_HEAD)/.venv/bin/activate`
    parser.add_argument(
        "--test-script",
        help="The script to run at each commit",
        required=True,
    )
    parser.add_argument(
        "--ignore-system-requirements",
        help="Ignores system requirements like Python 3.11 and tries to run even if they are not met.",
        action="store_true",
        default=False,
    )
    # TODO(scotttodd): --clean arg to `rm -rf` the workdir
    # TODO(scotttodd): control over logging
    #   redirect stdout/stderr from test script separate files in the workdir?

    return parser.parse_args()


def check_system_requirements(ignore_system_requirements):
    print("")
    system_check_okay = True

    # Check for Linux.
    print(
        f"  Current platform is '{platform.platform()}', platform.system is '{platform.system()}'."
    )
    if "Linux" not in platform.system():
        print("  ERROR! platform.system must be 'Linux'.", file=sys.stderr)
        system_check_okay = False

    # Check for Python 3.11.
    print("")
    print(f"  Current Python version is '{sys.version}'. This script requires 3.11.")
    if sys.version_info[:2] == (3, 11):
        python311_path = "python"
    else:
        python311_path = shutil.which("python3.11")
        if python311_path:
            print(f"  Found python3.11 at '{python311_path}', using that instead.")
        else:
            print(
                "  ERROR! Could not find Python version 3.11. Python version must be 3.11 to match package builds.",
                file=sys.stderr,
            )
            print(
                "  See `.github/workflows/pkgci_build_packages.yml` and `build_tools/pkgci/build_linux_packages.sh`.",
                file=sys.stderr,
            )
            system_check_okay = False

    # Check for 'gh'.
    print("")
    gh_path = shutil.which("gh")
    if not gh_path:
        print(
            "  ERROR! Could not find 'gh'. Install by following https://github.com/cli/cli#installation.",
            file=sys.stderr,
        )
        system_check_okay = False
    else:
        print(f"  Found gh at '{gh_path}'.")

    if not system_check_okay:
        print("")
        if ignore_system_requirements:
            print(
                "One or more configuration issues detected, but --ignore-system-requirements is set. Continuing.",
                file=sys.stderr,
            )
            return
        print(
            "One or more configuration issues detected. Fix the reported issues or pass --ignore-system-requirements to try running anyways. Exiting.",
            file=sys.stderr,
        )
        print("")
        print("------------------------------------------------------------------")
        sys.exit(1)

    return python311_path


def main(args):
    print("Welcome to bisect_packages.py!")

    print("")
    print("------------------------------------------------------------------")
    print("--------- Configuration ------------------------------------------")
    print("------------------------------------------------------------------")
    print("")
    print(f"  Searching range         : '{args.good_ref}' - '{args.bad_ref}'")

    print(f"  Using working directory : '{args.work_dir}'")
    Path.mkdir(args.work_dir, parents=True, exist_ok=True)

    print(f"  Using test script       : '{args.test_script}'")

    python311_path = check_system_requirements(args.ignore_system_requirements)

    print("")
    print("------------------------------------------------------------------")

    # Create new script in working directory that:
    #   * downloads the packages from the release and installs them
    #   * runs the original test script
    bisect_run_script = args.work_dir / "bisect_run_script.sh"
    with open(bisect_run_script, "w") as bisect_run_script_file:
        contents = ""
        contents += "#!/bin/bash\n"

        contents += "\n"
        contents += "#########################################\n"
        contents += "###### BISECT RELEASE SCRIPT SETUP ######\n"
        contents += "#########################################\n"
        contents += "\n"
        contents += "set -xeuo pipefail\n"
        contents += "\n"

        # Download packages for REF_HASH and install them into REF_HASH/.venv/.
        contents += "REF_HASH=$(git rev-parse BISECT_HEAD)\n"
        contents += f'"{python311_path}" '
        contents += str((THIS_DIR / ".." / "setup_venv.py").as_posix())
        contents += f" {args.work_dir}/"
        contents += "${REF_HASH}/.venv"
        contents += f" --artifact-path={args.work_dir}/"
        contents += "${REF_HASH} "
        contents += " --fetch-git-ref=${REF_HASH}\n"
        # Prepend the venv bin dir to $PATH. This is similar to running
        #   `source .venv/bin/activate`
        # while scoped to this process. Note that this does not modify
        # $PYTHONHOME or support the `deactivate` command.
        contents += f'PATH="{args.work_dir}/$REF_HASH/.venv/bin:$PATH"\n'

        contents += "\n"
        # Controlled failure - don't immediately exit. See below.
        contents += "set +e\n"
        contents += "\n"
        contents += "#########################################\n"
        contents += "############ ORIGINAL SCRIPT ############\n"
        contents += "#########################################\n"
        contents += "\n"

        with open(args.test_script, "r") as original_script:
            contents += original_script.read()

        contents += "\n"
        contents += "#########################################\n"
        contents += "##### BISECT RELEASE SCRIPT CLEANUP #####\n"
        contents += "#########################################\n"
        contents += "\n"
        # Controlled failure, See `set +e` above.
        # `git bisect` is looking for exit values in the 1-127 range, while
        # iree-compile can exit with value 245 sometimes:
        # https://git-scm.com/docs/git-bisect#_bisect_run. Here we just check
        # for non-zero and normalize back to 1.
        contents += "RET_VALUE=$?\n"
        contents += "if [ $RET_VALUE -ne 0 ]; then\n"
        contents += "    exit 1\n"
        contents += "fi\n"

        bisect_run_script_file.write(contents)

    os.chmod(str(bisect_run_script), 0o744)  # Set as executable.

    print("")
    print("------------------------------------------------------------------")
    print("--------- Running git bisect -------------------------------------")
    print("------------------------------------------------------------------")
    print("")
    subprocess.check_call(["git", "bisect", "reset"], cwd=REPO_ROOT)
    subprocess.check_call(
        [
            "git",
            "bisect",
            "start",
            # Just update the BISECT_HEAD reference instead of checking out the
            # ref for each iteration of the bisect process. We won't be building
            # from source and this script lives in the source tree, so keep the
            # repository in a stable state.
            # Note: scripts can access the hash via `git rev-parse BISECT_HEAD`.
            "--no-checkout",
            # We only care about the merge/aggregate commit when branches were
            # merged. Ignore ancestors of merge commits.
            "--first-parent",
        ],
        cwd=REPO_ROOT,
    )
    subprocess.check_call(["git", "bisect", "good", args.good_ref], cwd=REPO_ROOT)
    subprocess.check_call(["git", "bisect", "bad", args.bad_ref], cwd=REPO_ROOT)
    subprocess.check_call(
        ["git", "bisect", "run", str(bisect_run_script)], cwd=REPO_ROOT
    )

    print("")


if __name__ == "__main__":
    main(parse_arguments())
