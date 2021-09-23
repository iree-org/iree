#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Deletes the input number of tags and corresponding releases, starting with the
# oldest tags.

DELETE_TAGS_COUNT="$1"

if [[ -z "$DELETE_TAGS_COUNT" ]]; then
  echo "Must specify the number of tags to delete"
  exit 1
fi

TOTAL_TAGS_COUNT=$(git tag | wc -l)

if [[ "$DELETE_TAGS_COUNT" -ge "$TOTAL_TAGS_COUNT" ]]; then
  echo "Cannot delete all remaining tags"
  exit 1
fi

TAGS=$(git for-each-ref --sort=creatordate --count="$DELETE_TAGS_COUNT" --format '%(refname:short)' refs/tags)

git tag -d $TAGS
git push upstream -d $TAGS
