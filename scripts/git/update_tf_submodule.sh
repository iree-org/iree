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
# the TF submodule to the configured TENSORFLOW_COMMIT (default REMOTE for
# current HEAD).
#
# - Requries the gh CLI (https://github.com/cli/cli) to create a PR.
# - Will force push to the configured PR_BRANCH (default "tf-submodule-update")
#   on the configured FORK_REMOTE (default "fork")
# - Requires that local BASE_BRANCH branch is a pristine (potentially stale)
#   copy of the same branch on the configured UPSTREAM_REMOTE
#   (default "upstream").
# - Requires that the working directory be clean. Will abort otherwise.
# - An optional TF_COMMIT_NICKNAME nickname can be given to the commit for the
#   PR description. Otherwise, it will default to "current HEAD" if
#   TENSORFLOW_COMMIT is REMOTE and the trimmed commit sha otherwise.

set -e
set -x
set -o pipefail

TENSORFLOW_COMMIT="${1:-REMOTE}"
PR_BRANCH="tf-submodule-update"
BASE_BRANCH="${1:-google}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
FORK_REMOTE="${FORK_REMOTE:-fork}"
TF_COMMIT_NICKNAME=""

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working directory not clean. Aborting"
  git status
  exit 1
fi
if ! git symbolic-ref -q HEAD; then
  echo "In a detached HEAD state. Aborting"
  git status
  exit 1
fi
git checkout "${BASE_BRANCH?}"
git pull "${UPSTREAM_REMOTE?}" "${BASE_BRANCH?}" --ff-only
git submodule update --init
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working directory not clean after sync. Aborting"
  git status
  exit 1
fi
git checkout -B "${PR_BRANCH?}"

CMD="./scripts/git/update_tf_llvm_submodules.py --llvm_commit=KEEP --update_build_files=true --tensorflow_commit=${TENSORFLOW_COMMIT?}"

bash -c "${CMD?}"

TF_SHA="$(git submodule status third_party/tensorflow | awk '{print $1}' | cut -c -12)"

if [[ -z "${TF_COMMIT_NICKNAME?}" && "${TENSORFLOW_COMMIT?}" == "REMOTE" ]]; then
  TF_COMMIT_NICKNAME="current HEAD"
fi
TF_COMMIT_NICKNAME="${TF_COMMIT_NICKNAME:-${TF_SHA?}}"

TITLE="Integrate TF at tensorflow/tensorflow@${TF_SHA?}"
BODY="$(cat <<-EOF
Updates TF to
[${TF_COMMIT_NICKNAME?}](https://github.com/tensorflow/tensorflow/commit/${TF_SHA?})
and copies over the LLVM BUILD files.

\`${CMD?}\`
EOF
)"

git commit -am "${TITLE?}"
git push -f fork "${PR_BRANCH?}"
gh pr create --title="${TITLE?}" --body="${BODY?}" --base="${BASE_BRANCH?}"
