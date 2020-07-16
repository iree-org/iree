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

# Checks out a branch at the specified green commit (default "google") and
# creates a PR from that branch based on main.
#
# - Requries the gh CLI (https://github.com/cli/cli) to create a PR.
# - Will force push to the configured PR_BRANCH (default "google-to-main") on
#   the configured FORK_REMOTE (default "origin")
# - Requires that local "google" branch is a pristine (potentially stale) copy
#   of the "google" branch on the configured UPSTREAM_REMOTE
#   (default "upstream").
# - Requires that the working directory be clean. Will abort otherwise.

set -x
set -e
set -o pipefail

GREEN_COMMIT="${1:-google}"

PR_BRANCH="${PR_BRANCH:-google-to-main}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
FORK_REMOTE="${FORK_REMOTE:-origin}"

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
git checkout google
git pull "${UPSTREAM_REMOTE?}" google --ff-only
git submodule update --init
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working directory not clean after sync. Aborting"
  git status
  exit 1
fi
if [[ "${GREEN_COMMIT}" != "google" ]]; then
  git checkout "${GREEN_COMMIT?}"
  git submodule update
fi
git checkout -B "${PR_BRANCH?}"
git push -f "${FORK_REMOTE?}" "${PR_BRANCH?}"
if [[ -z "$(which gh)" ]]; then
  echo "gh not found on path."
  echo "Have you installed the GitHub CLI (https://github.com/cli/cli)?"
  echo "Cannot create PR. Branch ${BRANCH?} pushed, but aborting."
  exit 1
fi
gh pr create \
    --base main \
    --title="Merge google -> main" \
    --body="$(git log main.. --decorate=no --pretty='format:* %h %<(80,trunc)%s')"
