#!/usr/bin/env python3

# # Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Initialize git submodules for CI use.
# Here be dragons - NOT FOR REGULAR HUMAN DEVELOPMENT -
#
# Background:
#   * Updating git submodules involves fetching blobs of data from a server
#     (e.g. GitHub), unpacking/decompressing/resolving those blobs, and
#     creating a directory tree of files on the local disk. For large git
#     repositories (and submodules) like llvm/llvm-project, this can be a slow
#     process - particularly on certain types of machines or networks.
#   * Regular development workflows frequently move between commits in the git
#     history make edits to files, and run history-traversing commands. CI
#     workflows are much more limited in their access patterns and don't need
#     to persist the repository data after finishing.
#
# By limiting the number of git blobs we download and files we create, we can
# (hopefully) speed up the time it takes to initialize dependencies.
# Anecdotally, Windows CI machines on GitHub Actions can take upwards of 10
# minutes to clone llvm-project/ without these optimizations.
#
# Useful references:
#   https://github.blog/2020-12-21-get-up-to-speed-with-partial-clone-and-shallow-clone/
#   https://github.blog/2020-01-17-bring-your-monorepo-down-to-size-with-sparse-checkout/
#   https://github.com/Reedbeta/git-partial-submodule
#   https://git-scm.com/docs/git-sparse-checkout
#   https://llvm.org/docs/Proposals/GitHubMove.html#monorepo-variant

import os
import subprocess
import shutil
from typing import Sequence


def run_command(command: Sequence[str],
                **run_kwargs) -> subprocess.CompletedProcess:
  """Thin wrapper around subprocess.run"""
  print(f"-- Running: `{' '.join(command)}` --", flush=True)
  completed_process = subprocess.run(command,
                                     text=True,
                                     check=True,
                                     capture_output=False,
                                     stderr=subprocess.STDOUT,
                                     **run_kwargs)
  return completed_process


def sparse_clone_llvm(llvm_dir):
  print('-- Deleting third_party/llvm-project --', flush=True)
  shutil.rmtree(llvm_dir, ignore_errors=True)

  # Clone llvm-project/, without fetching any refs.
  run_command([
      'git', 'clone', '--depth=1', '--no-checkout',
      'https://github.com/iree-org/iree-llvm-fork.git',
      'third_party/llvm-project'
  ])

  # Set up sparse-checkout for just the directories we need to build IREE.
  # We're using "cone" mode (the default) here, which only lets us define a list
  # of directories to _include_. We could instead use the "non-cone" mode and
  # provide a list of .gitignore-style patterns. Non-cone mode would let us
  # exclude all test/ directories, for example.
  llvm_subdirs = [
      # Base dependencies
      'cmake/',
      'llvm/',
      'mlir/',
      # Extra dependencies needed for a few specific parts of the build
      'lld/',
      'libunwind/'
  ]
  run_command([
      'git',
      'sparse-checkout',
      'set',
  ] + llvm_subdirs, cwd=llvm_dir)


def get_llvm_submodule_hash():
  output = os.popen('git submodule status')
  submodules = output.readlines()
  for submodule in submodules:
    name = submodule.split()[1]
    if name != 'third_party/llvm-project':
      continue

    llvm_hash = submodule.split()[0].strip('-+')
    return llvm_hash

  raise RuntimeError("Could not find third_party/llvm-project submodule")


def run():
  llvm_dir = os.path.join(os.getcwd(), 'third_party/llvm-project')
  sparse_clone_llvm(llvm_dir)

  llvm_hash = get_llvm_submodule_hash()
  run_command(['git', 'fetch', '--depth=1', 'origin', llvm_hash], cwd=llvm_dir)
  run_command(['git', 'checkout', llvm_hash], cwd=llvm_dir)

  # Finish initializing all other submodules.
  run_command(
      ['git', 'submodule', 'update', '--init', '--jobs', '8', '--depth', '1'])


if __name__ == "__main__":
  run()
