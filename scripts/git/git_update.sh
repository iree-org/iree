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

# Checks out and updates the specified branch to match the corresponding branch
# on UPSTREAM_REMOTE (default "upstream")
#
# - Requires that the local branch is a pristine (potentially stale) copy of the
#   same branch on the configured UPSTREAM_REMOTE.
# - Requires that the working directory be clean. Will abort otherwise.

set -e
set -o pipefail

BRANCH="$1"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"

if [[ -z "$BRANCH" ]]; then
  echo "Must specify a branch to update for git_update.sh"
  exit 1
fi

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
git checkout "${BRANCH?}"
git pull "${UPSTREAM_REMOTE?}" "${BRANCH?}" --ff-only
git submodule update --init
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working directory not clean after update"
    git status
    exit 1
fi
