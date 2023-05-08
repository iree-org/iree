#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs yamllint on files modified vs the specified reference commit
# (default "main")

set -euo pipefail

BASE_REF="${1:-main}"

declare -a included_files_patterns=(
  "\.yaml$"
  "\.yml$"
)

declare -a excluded_files_patterns=(
  "/third_party/"
  "^third_party/"
  "/tools/test/"
  "^tools/test/"
)

# Join on |
included_files_pattern="$(IFS="|" ; echo "${included_files_patterns[*]?}")"
excluded_files_pattern="$(IFS="|" ; echo "${excluded_files_patterns[*]?}")"

readarray -t files < <(\
  (git diff --name-only --diff-filter=d "${BASE_REF}" || kill $$) \
    | grep -E "${included_files_pattern?}" \
    | grep -v -E "${excluded_files_pattern?}")

if (( ${#files[@]} == 0 )); then
  echo "No Yaml files changed"
  exit 0
fi


yamllint --strict "${files[@]}"
