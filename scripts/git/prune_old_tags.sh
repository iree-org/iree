#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Deletes all but the input number of snapshot tags and corresponding releases,
# starting with the oldest tags.
#
# Delete all but the 100 newest tags from the repository at the UPSTREAM_REMOTE
# environment variable:
#   ./prune_old_tags.sh 100
#
# List the tags that would be deleted when not performing a dry run:
#   ./prune_old_tags.sh 100 DRY_RUN

set -euo pipefail

KEEP_TAGS_COUNT="$1"
DRY_RUN="${2:-}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"

DRY_RUN="${DRY_RUN//[-_]/}"
DRY_RUN="${DRY_RUN^^}"

if [[ -n "${DRY_RUN}" ]] && [[ "${DRY_RUN}" != "DRYRUN" ]]; then
  echo "Unexpected argument '${2}'. Should be DRYRUN (case and punctuation insensitive) if present"
  exit 1
fi

if [[ -z "${KEEP_TAGS_COUNT}" ]]; then
  echo "Must specify the number of tags to keep"
  exit 1
fi

if [[ "${KEEP_TAGS_COUNT}" -le 0 ]]; then
  echo "Cannot delete all remaining tags"
  exit 1
fi

git fetch --all --tags
TAGS=($(git for-each-ref --sort=creatordate --format '%(refname:short)' refs/tags))

# Filter out tags that were not created by snapshot releases.
for TAG_INDEX in "${!TAGS[@]}"; do
  if [[ ! ${TAGS[TAG_INDEX]} =~ ^snapshot-* ]]; then
    unset 'TAGS[TAG_INDEX]'
  fi
done

TOTAL_TAGS_COUNT=${#TAGS[@]}
if [[ "${KEEP_TAGS_COUNT}" -ge "${TOTAL_TAGS_COUNT}" ]]; then
  echo "Only ${TOTAL_TAGS_COUNT} tags exist, so nothing to delete"
  exit 1
fi

DELETE_TAGS_COUNT=$((TOTAL_TAGS_COUNT-KEEP_TAGS_COUNT))
DELETE_TAGS=${TAGS[@]::$DELETE_TAGS_COUNT}

echo "${TOTAL_TAGS_COUNT} snapshot tags available"
echo "${DELETE_TAGS_COUNT} tags will be deleted"

if [[ "${DRY_RUN}" == "DRYRUN" ]]; then
  echo "Dry run mode, these tags would be deleted:"
  for TAG in ${DELETE_TAGS[@]}; do
    echo "${TAG}"
  done
  exit 0
fi

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! "${REPLY}" =~ ^[Yy]$ ]]
then
  echo "Exiting"
  exit 0
fi

git tag -d ${DELETE_TAGS[@]}
git push "${UPSTREAM_REMOTE?}" -d ${DELETE_TAGS[@]}
