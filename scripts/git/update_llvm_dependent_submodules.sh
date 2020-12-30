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

# Creates a PR based on the specified BASE_BRANCH (default "google") to update
# the LLVM-dependent submodules to the configured commits.
#
# Positional args will be passed through to update_to_llvm_syncpoint.py
#
# - Requries the gh CLI (https://github.com/cli/cli) to create a PR.
# - Will force push to the configured PR_BRANCH (default
#   "llvm-dependent-submodule-update") on the configured FORK_REMOTE (default
#   "origin")
# - Requires that local BASE_BRANCH branch is a pristine (potentially stale)
#   copy of the same branch on the configured UPSTREAM_REMOTE
#   (default "upstream").
# - Requires that the working directory be clean. Will abort otherwise.

set -e
set -o pipefail

export UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
PR_BRANCH="llvm-dependent-submodule-update"
BASE_BRANCH="${BASE_BRANCH:-google}"
FORK_REMOTE="${FORK_REMOTE:-origin}"

./scripts/git/git_update.sh "${BASE_BRANCH?}"
git checkout -B "${PR_BRANCH?}"

CMD="./scripts/git/update_to_llvm_syncpoint.py $@"

bash -c "${CMD?}"

LLVM_SHA="$(git submodule status third_party/llvm-project | awk '{print $1}' | cut -c -12)"
LLVM_BAZEL_SHA="$(git submodule status third_party/llvm-bazel | awk '{print $1}' | cut -c -12)"
TF_SHA="$(git submodule status third_party/tensorflow | awk '{print $1}' | cut -c -12)"
MLIR_HLO_SHA="$(git submodule status third_party/mlir_hlo | awk '{print $1}' | cut -c -12)"

TITLE="Synchronize submodules with LLVM at llvm/llvm-project@${LLVM_SHA?}"
BODY="$(cat <<-EOF
Updates LLVM dependencies to match
[${LLVM_SHA?}](https://github.com/llvm/llvm-project/commit/${LLVM_SHA?}).
- llvm-bazel to
  [${LLVM_BAZEL_SHA?}](https://github.com/google/llvm-bazel/commit/${LLVM_BAZEL_SHA?})
- TensorFlow to
  [${TF_SHA?}](https://github.com/tensorflow/tensorflow/commit/${TF_SHA?})
- MLIR-HLO to
  [${MLIR_HLO_SHA?}](https://github.com/tensorflow/mlir-hlo/commit/${MLIR_HLO_SHA?})

\`${CMD?}\`
EOF
)"

git commit -am "${TITLE?}"
git push -f "${FORK_REMOTE?}" "${PR_BRANCH?}"

if [[ -z "$(which gh)" ]]; then
  echo "gh not found on path."
  echo "Have you installed the GitHub CLI (https://github.com/cli/cli)?"
  echo "Cannot create PR. Branch ${PR_BRANCH?} pushed, but aborting."
  echo "You can manually create a PR using the generated body:"
  echo "${BODY?}"
  exit 1
fi

# Workaround https://github.com/cli/cli/issues/1820
GITHUB_USERNAME="$(gh config get -h github.com user)"
gh pr create --base="${BASE_BRANCH?}" --head="${GITHUB_USERNAME?}:${PR_BRANCH?}" --title="${TITLE?}" --body="${BODY?}"
