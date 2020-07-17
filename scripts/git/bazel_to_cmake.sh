#!/bin/bash

# Copyright 2020 Google LLC
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

# Creates a PR based on the specified BASE_BRANCH (default "google") to fix
# CMake files based on Bazel BUILD files.
#
# - Requries the gh CLI (https://github.com/cli/cli) to create a PR.
# - Will force push to the configured PR_BRANCH (default "bazel-to-cmake-fix")
#   on the configured FORK_REMOTE (default "origin")
# - Requires that local BASE_BRANCH branch is a pristine (potentially stale)
#   copy of the same branch on the configured UPSTREAM_REMOTE
#   (default "upstream").
# - Requires that the working directory be clean. Will abort otherwise.

set -e
set -o pipefail

export UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
BASE_BRANCH="${1:-google}"
PR_BRANCH="bazel-to-cmake-fix"
FORK_REMOTE="${FORK_REMOTE:-origin}"

./scripts/git/git_update.sh "${BASE_BRANCH?}"
git checkout -B "${PR_BRANCH?}"
./build_tools/bazel_to_cmake/bazel_to_cmake.py

TITLE="Run Bazel -> CMake"
BODY='Updates CMake files to match Bazel BUILD

\`./build_tools/bazel_to_cmake/bazel_to_cmake.py\`
'

git commit -am "${TITLE?}"
git push -f "${FORK_REMOTE}" "${PR_BRANCH?}"

if [[ -z "$(which gh)" ]]; then
  echo "gh not found on path."
  echo "Have you installed the GitHub CLI (https://github.com/cli/cli)?"
  echo "Cannot create PR. Branch ${PR_BRANCH?} pushed, but aborting."
  echo "You can manually create a PR using the generated body:"
  echo "${BODY?}"
  exit 1
fi

gh pr create --title="${TITLE?}" --body="${BODY?}" --base="${BASE_BRANCH?}"
