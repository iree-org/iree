#!/usr/bin/env python
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from datetime import date
import sys
import textwrap

import iree_utils

LLVM_REPO_DIR = iree_utils.get_submodule_root("llvm-project")
TRACK_PATHS = ("mlir",)


class CurrentState:
    """Current state of the llvm-project integrate."""

    def __init__(self, args):
        self.current_commit, self.current_summary = iree_utils.git_current_commit(
            repo_dir=LLVM_REPO_DIR
        )
        # The common commit between the llvm-project submodule and upstream.
        self.merge_base_commit = iree_utils.git_merge_base(
            self.current_commit, "upstream/main", repo_dir=LLVM_REPO_DIR
        )
        # Whether the current llvm-project commit is clean (True) or
        # carries patches (False).
        self.is_clean = self.merge_base_commit == self.current_commit
        # List of (commit, desc) tuples in reverse chronological order for
        # commits that upstream is ahead.
        self.new_commits = iree_utils.git_log_range(
            refs=("upstream/main", f"^{self.merge_base_commit}"),
            paths=TRACK_PATHS,
            repo_dir=LLVM_REPO_DIR,
        )


def do_start(args):
    fetch(args)
    state = CurrentState(args)
    if not state.is_clean:
        raise RuntimeError("Current branch state is unclean. Not implemented yet.")
    if not state.new_commits:
        print(f"Up to date! Not starting.")
        return

    next_commit, next_desc = next(reversed(state.new_commits))
    print(f"==> Starting new integrate")
    # Create branch.
    branch_name = args.branch_name
    if not branch_name:
        branch_name = f"increment-llvm-{date.today().strftime('%Y%m%d')}"
    print(f"  Creating branch {branch_name} (override with --branch-name=)")
    iree_utils.git_create_branch(
        branch_name,
        checkout=True,
        ref="HEAD",
        force=args.reuse_branch,
    )
    iree_utils.git_reset(next_commit, repo_dir=LLVM_REPO_DIR)
    iree_utils.git_create_commit(
        message=(
            f"Start LLVM integrate ({len(state.new_commits)} commits behind)\n\n"
            f"Advance LLVM to {next_commit}: {next_desc}"
        ),
        add_all=True,
    )
    print("Pushing...")
    iree_utils.git_push_branch("origin", branch_name, force=args.reuse_branch)


def do_status(args):
    fetch(args)
    state = CurrentState(args)
    print(f"==> llvm-project is currently at {state.current_summary}:")
    if state.is_clean:
        print(f"  : Current commit is clean (no patches)")
    else:
        # TODO: Also get the merge base with --independent to get the carried
        # patches.
        print(
            f"  : Current commit has diverging patches with base {state.merge_base_commit}"
        )

    # Compute the different commits.
    print(
        f"==> {len(state.new_commits)} affecting commits between upstream head and current:"
    )
    for commit, desc in state.new_commits:
        print(f"  {commit}: {desc}")


def fetch(args):
    print("==> Fetching origin and upstream revisions...")
    setup_remotes(args)
    iree_utils.git_fetch(repository="origin")
    iree_utils.git_fetch(repository="origin", repo_dir=LLVM_REPO_DIR)
    iree_utils.git_fetch(repository="upstream", repo_dir=LLVM_REPO_DIR)


def setup_remotes(args):
    # We need to know what the real upstream repo is.
    iree_utils.git_setup_remote(
        "upstream", "https://github.com/llvm/llvm-project.git", repo_dir=LLVM_REPO_DIR
    )


def main(args):
    if args.sub_command == "start":
        do_start(args)
    elif args.sub_command == "status":
        do_status(args)
    else:
        raise ValueError(f"Unrecognized sub-command {args.sub_command}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="IREE LLVM-bump-inator")
    subparsers = parser.add_subparsers(
        help="sub-command help", required=True, dest="sub_command"
    )
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument(
        "--branch-name", help="Integrate branch to create", default=None
    )
    start_parser.add_argument(
        "--reuse-branch",
        help="Allow re-use of an existing branch",
        action="store_true",
        default=False,
    )
    status_parser = subparsers.add_parser("status")

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
