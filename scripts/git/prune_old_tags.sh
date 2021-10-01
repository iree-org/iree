#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Deletes all but the input number of tags and corresponding releases, starting
# with the oldest tags.

KEEP_TAGS_COUNT="$1"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$KEEP_TAGS_COUNT" ]]; then
  echo "Must specify the number of tags to keep"
  exit 1
fi

if [[ "$KEEP_TAGS_COUNT" -le 0 ]]; then
  echo "Cannot delete all remaining tags"
  exit 1
fi

git fetch --all --tags
TOTAL_TAGS_COUNT=$(git tag | wc -l)

if [[ "$KEEP_TAGS_COUNT" -ge "$TOTAL_TAGS_COUNT" ]]; then
  echo "Only $TOTAL_TAGS_COUNT exist, so nothing to delete"
  exit 1
fi

DELETE_TAGS_COUNT=$((TOTAL_TAGS_COUNT-KEEP_TAGS_COUNT))
TAGS=($(git for-each-ref --sort=creatordate --count="$DELETE_TAGS_COUNT" --format '%(refname:short)' refs/tags))

# Filter out tags that were not created by snapshot releases.
for TAG_INDEX in "${!TAGS[@]}"; do
  if [[ ! ${TAGS[TAG_INDEX]} =~ ^snapshot-* ]]; then
    unset 'TAGS[TAG_INDEX]'
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run mode: these tags would be deleted"
  for TAG in ${TAGS[@]}; do
    echo $TAG
  done
  exit 1
fi

git tag -d ${TAGS[@]}
git push "${UPSTREAM_REMOTE?}" -d ${TAGS[@]}
