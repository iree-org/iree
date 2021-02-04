#!/bin/bash
# Copyright 2021 Google LLC
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

# Fixes a Copybara push that failed to create a merge commit, using the
# COPYBARA_TAG label to add a second parent to the HEAD commit.

set -e

COPYBARA_TAG="COPYBARA_INTEGRATE_REVIEW"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working directory not clean. Aborting"
    git status
    exit 1
fi

# Get the commit message for the head commit
MESSAGE="$(git log --format=%B -n 1 HEAD)"

# Extract the commit to merge from using the Copybara tag.
MERGE_FROM="$(echo "${MESSAGE?}" | awk -v pat="${COPYBARA_TAG?}" '$0~pat{print $NF}')"

# And create a new message with the tag removed
NEW_MESSAGE="$(echo "${MESSAGE?}" | sed "/${COPYBARA_TAG?}/d")"

git fetch "${UPSTREAM_REMOTE?}" "${MERGE_FROM?}"
git reset --soft "$(git commit-tree -m "${NEW_MESSAGE?}" -p HEAD^ -p ${MERGE_FROM?} HEAD^{tree})"
