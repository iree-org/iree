#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks out a branch at the specified green commit (default "main") and
# creates a PR from that branch based on google.
#
# - Requries the gh CLI (https://github.com/cli/cli) to create a PR.
# - Will force push to the configured PR_BRANCH (default "main-to-google") on
#   the configured FORK_REMOTE (default "origin")
# - Requires that local "main" branch is a pristine (potentially stale) copy
#   of the "main" branch on the configured UPSTREAM_REMOTE
#   (default "upstream").
# - Requires that the working directory be clean. Will abort otherwise.

set -e
set -o pipefail

GREEN_COMMIT="${1:-main}"

export UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
PR_BRANCH="${PR_BRANCH:-main-to-google}"
FORK_REMOTE="${FORK_REMOTE:-origin}"

./scripts/git/git_update.sh main
if [[ "${GREEN_COMMIT}" != "main" ]]; then
  git checkout "${GREEN_COMMIT?}"
  git submodule update
fi

git checkout -B "${PR_BRANCH?}"
git push -f "${FORK_REMOTE?}" "${PR_BRANCH?}"

TITLE="Merge main -> google"

git fetch "${UPSTREAM_REMOTE?}" google
BODY="$(./scripts/git/summarize_changes.sh ${UPSTREAM_REMOTE?}/google)"

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
gh pr create --base google --head="${GITHUB_USERNAME?}:${PR_BRANCH?}" --title="${TITLE?}" --body="${BODY?}"
