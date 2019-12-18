#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Updates the LLVM, MLIR, and TF submodules and pushes a commit to master.

# This is scripted because we have to do this all the time as we get out of
# sync between GitHub and upstream. It's not a more sophisticated fix
# because we're waiting until MLIR moves to LLVM to do something more
# robust here.

set -e

current_branch="$(git branch --show-current)"

if [ "$current_branch" != "master" ]; then
  echo "Current branch (${current_branch}) is not master. Aborting."
  exit 1
fi

git_status="$(git status -s)"

if [ -n "$git_status" ]; then
  echo "Modified/untracked files in working directory. Aborting."
  exit 1
fi

upstream_url="https://github.com/google/iree.git"
echo "Pulling latest from remote ${upstream_url?}"
git pull ${upstream_url}

git_status="$(git status -s)"

if [ -n "$git_status" ]; then
  echo "Modified/untracked files in working directory after pull. Aborting."
  exit 1
fi

echo "Updating LLVM, MLIR, TF submodules to latest"

git submodule update --remote third_party/llvm-project third_party/mlir third_party/tensorflow

MSG="Update LLVM, MLIR, TF submodules to latest

\$ ./build_tools/scripts/update_submodules.sh
"

git commit -am "$MSG"

echo -e "\n\n\nCreated commit. Please confirm message and diffs before pushing"

git log -n 1
git diff origin/master
