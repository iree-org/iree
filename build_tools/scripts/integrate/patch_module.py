#!/usr/bin/env python
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Pushes module patches to an appropriate repository.
#
# Typical steps:
# 1. Advance the main branches on our forks to get all upstream commits
#    (if you forget to do this, your cherry-pick will likely complain about
#    a bad ref).
# 2. In your submodule:
#      git fetch
#      git cherry-pick <some commit>
# 3. Run this script from the main IREE repository (one of the following):
#      patch_module.py --module=llvm-project
#      patch_module.py --module=mlir-hlo
# 4. Send a PR on the main IREE repo to bump the submodule. Be sure to include
#    the name of the patch branch for posterity.

import argparse
from datetime import date
import os
import sys

import iree_utils
import iree_modules

PATCH_REMOTE_ALIAS = "patched"


def main(args):
    module_info = iree_modules.MODULE_INFOS.get(args.module)
    if not module_info:
        raise SystemExit(f"ERROR: Bad value for --module. Must be one of: "
                         f"{', '.join(iree_modules.MODULE_INFOS.keys())}")

    if args.command == "patch":
        main_patch(args, module_info)
    else:
        raise SystemExit(
            f"ERROR: Unrecognized --command. Must be one of: patch, unpatch")


def main_patch(args, module_info: iree_modules.ModuleInfo):
    module_root = os.path.join(iree_utils.get_repo_root(), module_info.path)
    setup_module_remotes(module_root, module_info)

    branch_name = find_unused_branch_name(module_info)
    print(f"Allocated branch: {branch_name}")
    current_commit, summary = iree_utils.git_current_commit(
        repo_dir=module_root)
    print(f"Module is currently at: {summary}")
    print(
        f"*** Pushing branch {branch_name} to {module_info.fork_repository_push} ***"
    )
    print(f"(Please ignore any messages below about creating a PR)\n")
    iree_utils.git_exec([
        "push", PATCH_REMOTE_ALIAS,
        f"{current_commit}:refs/heads/{branch_name}"
    ],
                        repo_dir=module_root)
    print(f"*** Branch {branch_name} pushed ***")

    print(f"******* Congratulations *******")
    print(f"You have pushed your commits to {branch_name} on {module_info.fork_repository_push}.")
    print(f"Your main repository should now show that the submodule has been edited.")
    print(f"Make a commit, referencing the above branch cherry-picks and ")
    print(f"land the resulting PR.")
    print(f"You can push more commits to this module's patch branch via:")
    print(f"  (cd {module_info.path} && git push {PATCH_REMOTE_ALIAS} HEAD:{branch_name})")


def setup_module_remotes(module_root: str,
                         module_info: iree_modules.ModuleInfo):
    iree_utils.git_setup_remote(PATCH_REMOTE_ALIAS,
                                url=module_info.fork_repository_push,
                                repo_dir=module_root)


def find_unused_branch_name(module_info: iree_modules.ModuleInfo):
    branch_base = f"{module_info.branch_prefix}{date.today().strftime('%Y%m%d')}"
    branch_name = branch_base
    existing_branches = iree_utils.git_ls_remote_branches(
        module_info.fork_repository_pull,
        filter=[f"refs/heads/{module_info.branch_prefix}*"])
    i = 1
    while branch_name in existing_branches:
        branch_name = f"{branch_base}.{i}"
        i += 1
    return branch_name


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="IREE Submodule Patcher")
    parser.add_argument("--module",
                        help="Submodule to operate on",
                        default=None)
    parser.add_argument("--command",
                        help="Command to execute",
                        default="patch")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
