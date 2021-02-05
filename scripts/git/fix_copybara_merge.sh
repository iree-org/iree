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
# COPYBARA_TAG label to add a second parent to the HEAD commit. This should be
# run when Copybara exports a 'main -> google' commit, but fails to create a
# merge commit. The failure to create such a commit means that the
# COPYBARA_INTEGRATE_REVIEW tag is left in the commit message. It should only
# be run on the google branch. After running this script, you can verify the git
# log looks as expected, using something like:
#
# git log --left-right --graph --oneline --boundary google...main
#
# and then force push over the google branch. Force pushing is destructive. If
# you're uncertain, ask!

set -e

COPYBARA_TAG="${COPYBARA_TAG:-COPYBARA_INTEGRATE_REVIEW}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"

# Get the commit message of the HEAD commit
MESSAGE="$(git log --format=%B -n 1 HEAD)"

################################ Safety checks ################################

if [[ -n "$(git status --porcelain)" ]]; then
  echo -e "\n\nWorking directory not clean. Aborting"
  git status
  exit 1
fi

if ! echo "${MESSAGE?}" | grep -q "${COPYBARA_TAG}"; then
  echo -e "\n\nHEAD commit does not contain Copybara tag '${COPYBARA_TAG?}'. Aborting."
  git log -n 1 HEAD
  exit 1
fi

# Technically this works anywhere, but our only current use case is to run it on
# the google branch and this is a weird and destructive change, so just be
# really picky about it.
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH?}" != "google" ]]; then
  echo -e "\n\nCurrent branch ${CURRENT_BRANCH?} is not 'google'. Aborting"
  exit 1
fi

# We don't want to be rewriting commits that have already hit the main branch.
# Obviously this is not foolproof because there can be a race, but still not a
# bad check to have.
git fetch "${UPSTREAM_REMOTE?}" main:main
if git merge-base --is-ancestor HEAD main; then
  echo -e "\n\nHEAD commit is already on main branch. Aborting"
  exit 1
fi

################################################################################

echo -e "\n\nTo revert the changes made by this script, run:"
echo "git reset --hard $(git rev-parse HEAD)"

# Fix submodules
./scripts/git/submodule_versions.py import

if ! git diff --cached --exit-code; then
  echo -e "\n\nUpdating commit with fixed submodules"
  git commit --amend -a --no-edit
fi

COPYBARA_LINE="$(echo "${MESSAGE?}" | grep "${COPYBARA_TAG?}")"

# Extract the commit to merge from using the Copybara tag.
MERGE_FROM="$(echo "${COPYBARA_LINE?}" | awk '{print $NF}')"

# Extract the PR URL from the Copybara tag
PR_URL="$(echo "${COPYBARA_LINE?}" | awk -F'[= ]' '{print $2}')"

if [[ -z "${MERGE_FROM?}" ]]; then
  echo -e "\n\nFailed extracting commit to merge from. Aborting"
  exit 1
fi

# Create a new message with the tag removed
NEW_MESSAGE="$(echo "${MESSAGE?}" | sed "/${COPYBARA_TAG?}/d")"

# Make sure we actually have the commit we need to merge from.
echo -e "\n\nFetching ${MERGE_FROM?}"
git fetch "${UPSTREAM_REMOTE?}" "${MERGE_FROM?}"

echo -e "\n\nIdentified ${MERGE_FROM?} as commit to merge from:"
git log -n 1 "${MERGE_FROM?}"

if [[ -z "$(git rev-list --merges HEAD^..HEAD)" ]]; then
  # Add a tag to the commit to merge from so it is highlighted in the git log.
  # If someone knows how to just highlight an individual commit with git log,
  # that would be preferable.
  git tag "merge-from-${MERGE_FROM?}" "${MERGE_FROM?}"

  echo -e "\n\nCurrent git log graph:"
  git log --left-right --graph --oneline --boundary "HEAD...main"

  # Create a new commit object `git commit-tree` based on the tree of the
  # current HEAD commit with the parent of the HEAD commit as first parent and
  # the commit to merge from as the second. Use the new message as the commit
  # message. Reset the current branch to this commit.
  # See https://stackoverflow.com/q/48560351
  git reset --soft "$(git commit-tree -m "${NEW_MESSAGE?}" -p HEAD^ -p ${MERGE_FROM?} HEAD^{tree})"

  echo -e "\n\nCreated fake merge. New git log graph:"
  git log --left-right --graph --oneline --boundary "HEAD...main"

  # Delete the tag we created
  git tag -d "merge-from-${MERGE_FROM?}"
  exit 0
fi

echo -e "\n\nHEAD commit is already a merge commit. Will not create a new merge."
# Just rewrite the commit message.
git commit --amend --no-edit --message="${NEW_MESSAGE?}"
if [[ -z "$(which gh)" ]]; then
  echo "gh not found on path."
  echo "Have you installed the GitHub CLI (https://github.com/cli/cli)?"
  echo "Cannot close PR ${PR_URL?}"
  echo "You can manually close the PR from the GitHub UI."
else
  # Technically this commit doesn't exist in the repository yet, but the
  # alternatives are to:
  # 1. Not close the PR automatically -> clutter in the repo and manual work.
  # 2. Force-push automatically as part of this script. Fine for use in the
  #    GitHub action, but humans will probably want to check everything first.
  # 3. Reparse the PR URL as part of the action and close it there. Duplicated
  #    logic that is likely to get out of sync.
  # 4. Propagate the PR URL back to the calling action. Not sure how to do
  #    that and involves coordinating moving parts.
  echo "Closing ${PR_URL?}"
  gh pr comment --body "Closed by $(git rev-parse HEAD)"
  gh pr close -d "${PR_URL?}"
fi
