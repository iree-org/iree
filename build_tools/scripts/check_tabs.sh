#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks for tabs in files modified vs the specified reference commit (default
# "main")

set -o pipefail

BASE_REF="${1:-main}"

declare -a excluded_files_patterns=(
  "^\.gitmodules$"
  "/third_party/"
  "^third_party/"
  "*Makefile*"
  # Symlinks make grep upset
  "^integrations/tensorflow/iree-dialects$"
  # Generated / Binary files
  ".onnx"
  ".svg"
)

# Join on |
excluded_files_pattern="$(IFS="|" ; echo "${excluded_files_patterns[*]?}")"

readarray -t files < <(\
  (git diff --name-only --diff-filter=d "${BASE_REF}" || kill $$) \
    | grep -v -E "${excluded_files_pattern?}")

if (( ${#files[@]} == 0 )); then
  exit 0
fi;

diff="$(grep --with-filename --line-number --perl-regexp --binary-files=without-match '\t' "${files[@]}")"

grep_exit="$?"

if (( "${grep_exit?}" >= 2 )); then
  exit "${grep_exit?}"
fi

if (( "${grep_exit?}" == 0 )); then
  echo "Changed files include tabs. Please use spaces.";
  echo "$diff"
  exit 1;
fi
