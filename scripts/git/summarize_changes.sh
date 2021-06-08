#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Creates a commit summary of commits between the specified refs.
#
# Get changes between the current HEAD commit and the latest fetch of the
# upstream repository main branch:
#   summarize_changes.sh upstream/main
#   summarize_changes.sh upstream/main HEAD
#
# Summarize commits between the local main-to-google branch and the latest fetch
# of the upstream repository main branch
#   summarize_changes.sh upstream/main main-to-google

set -e
set -o pipefail

BASE_REF="${1}"
NEW_REF="${2:-HEAD}"

# Print commits with their short hash and the first 80 characters of their
# commit title. Use awk to trim the trailing whitespace introduced by git log

git log \
  "${BASE_REF?}..${NEW_REF?}" \
  --first-parent \
  --decorate=no \
  --pretty='format:* %h %<(80,trunc)%s' \
  | awk '{$1=$1;print}'
