#!/usr/bin/env python
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Creates a new branch that bumps the llvm-project commit.
# Typical usage (from the iree/ repository):
#   /path/to/here/bump_llvm.py
#
# In the default configuration, it will create a new branch
#   "bump-llvm-YYYYMMDD"
# This will fail if the branch already exists, in which case, you can:
#   * Specify an explicit branch name with --branch-name=my-integrate
#   * Pass "--reuse-branch" if you are sure that you want to lose the current
#     branch state. This is largely meant for developing this script or YOLO
#     use.
#
# In order to not interfere with your personal preferences, a remote named
# 'UPSTREAM_AUTOMATION' is used (and setup if needed). Sorry if you use that
# name for your daily work.
#
# This then reverts any changes to the llvm-project submodule setup (i.e.
# resets it to llvm-project's repository and disables branch tracking) and
# resets the submodule to the curren HEAD commit, generating a nice commit
# message.
#
# The branch is then pushed to the main repository. GitHub will print the usual
# message to create a pull request, which you should do to kick off pre-merge
# checks. You should land changes to this branch until green (or get other
# people to do so).
#
# When satisfied, Squash and Merge, opting to delete the branch to keep things
# tidy.

import argparse
from datetime import date
import os
import sys

import iree_modules
import iree_utils


def main(args):
    if not args.disable_setup_remote:
        iree_utils.git_setup_remote(args.upstream_remote, args.upstream_repository)

    iree_utils.git_check_porcelain()
    print(f"Fetching remote repository: {args.upstream_remote}")
    iree_utils.git_fetch(repository=args.upstream_remote)

    # If re-using a branch, make sure we are not on that branch.
    if args.reuse_branch:
        iree_utils.git_checkout("main")

    # Create branch.
    branch_name = args.branch_name
    if not branch_name:
        branch_name = f"bump-llvm-{date.today().strftime('%Y%m%d')}"
    print(f"Creating branch {branch_name} (override with --branch-name=)")
    iree_utils.git_create_branch(
        branch_name,
        checkout=True,
        ref=f"{args.upstream_remote}/main",
        remote=f"{args.upstream_remote}",
        force=args.reuse_branch,
    )

    # Reset the llvm-project submodule to track upstream.
    # This will discard any cherrypicks that may have been committed locally,
    # but the assumption is that if doing a main llvm version bump, the
    # cherrypicks will be incorporated at the new commit. If not, well, ymmv
    # and you will find out.
    iree_utils.git_submodule_set_origin(
        "third_party/llvm-project",
        url="https://github.com/shark-infra/llvm-project.git",
        branch="--default",
    )

    # Remove the branch pin file, reverting us to pure upstream.
    branch_pin_file = os.path.join(
        iree_utils.get_repo_root(),
        iree_modules.MODULE_INFOS["llvm-project"].branch_pin_file,
    )
    if os.path.exists(branch_pin_file):
        os.remove(branch_pin_file)

    # Update the LLVM submodule.
    llvm_commit = args.llvm_commit
    print(f"Updating LLVM submodule to {llvm_commit}")
    llvm_root = iree_utils.get_submodule_root("llvm-project")
    iree_utils.git_fetch(repository="origin", ref="refs/heads/main", repo_dir=llvm_root)
    if llvm_commit == "HEAD":
        llvm_commit = "origin/main"
    iree_utils.git_reset(llvm_commit, repo_dir=llvm_root)
    llvm_commit, llvm_summary = iree_utils.git_current_commit(repo_dir=llvm_root)
    print(f"LLVM submodule reset to:\n  {llvm_summary}\n")

    # Create a commit.
    print("Create commit...")
    iree_utils.git_create_commit(
        message=(
            f"Integrate llvm-project at {llvm_commit}\n\n"
            f"* Reset third_party/llvm-project: {llvm_summary}"
        ),
        add_all=True,
    )

    # Push.
    print("Pushing...")
    iree_utils.git_push_branch(args.upstream_remote, branch_name)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="IREE LLVM-bump-inator")
    parser.add_argument(
        "--upstream-remote", help="Upstream remote", default="UPSTREAM_AUTOMATION"
    )
    parser.add_argument(
        "--upstream-repository",
        help="Upstream repository URL",
        default="git@github.com:openxla/iree.git",
    )
    parser.add_argument(
        "--disable-setup-remote",
        help="Disable remote setup",
        action="store_true",
        default=False,
    )
    parser.add_argument("--llvm-commit", help="LLVM commit sha", default="HEAD")
    parser.add_argument(
        "--branch-name", help="Integrate branch to create", default=None
    )
    parser.add_argument(
        "--reuse-branch",
        help="Allow re-use of an existing branch",
        action="store_true",
        default=False,
    )
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
