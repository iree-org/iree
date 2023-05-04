#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs all the lint checks that we run on GitHub locally. Skips checks if the
# relevant tool doesn't exist.

# Keep this in sync with .github/workflows/lint.yml

# WARNING: this script *makes changes* to the working directory and the index.

set -uo pipefail

FINAL_RET=0
LATEST_RET=0

scripts_dir="$(dirname $0)"

function update_ret() {
  LATEST_RET="$?"
  if [[ "${LATEST_RET}" -gt "${FINAL_RET}" ]]; then
    FINAL_RET="${LATEST_RET}"
  fi
}

# Update the exit code after every command
function enable_update_ret() {
  trap update_ret DEBUG
}

function disable_update_ret() {
  trap - DEBUG
}

function exists() {
  command -v "${1}" > /dev/null
}


echo "***** Uncommitted changes *****"
git add -A
git diff HEAD --exit-code

if [[ $? -ne 0 ]]; then
  echo "Found uncomitted changes in working directory. This script requires" \
        "all changes to be committed. All changes have been added to the" \
        "index. Please commit or clean all changes and try again."
  exit 1
fi

enable_update_ret

echo "***** Bazel -> CMake *****"
./build_tools/bazel_to_cmake/bazel_to_cmake.py
./build_tools/bazel_to_cmake/bazel_to_cmake.py --root_dir=integrations/tensorflow/e2e
git add -A
git diff HEAD --exit-code
trap - DEBUG

echo "***** buildifier *****"
# Don't fail script if condition is false
disable_update_ret
if exists buildifier; then
  enable_update_ret
  ${scripts_dir}/run_buildifier.sh
  git diff --exit-code
else
  enable_update_ret
  echo "'buildifier' not found. Skipping check"
fi

echo "***** yapf *****"
# Don't fail script if condition is false
disable_update_ret
if exists yapf; then
  enable_update_ret
  git diff -U0 main | ./third_party/format_diff/format_diff.py yapf -i
else
  enable_update_ret
  echo "'yapf' not found. Skipping check"
fi

echo "***** pytype *****"
# Don't fail script if condition is false
disable_update_ret
if exists pytype; then
  enable_update_ret
  ./build_tools/pytype/check_diff.sh
else
  enable_update_ret
  echo "'pytype' not found. Skipping check"
fi

echo "***** clang-format *****"
# Don't fail script if condition is false
disable_update_ret
if exists git-clang-format; then
  enable_update_ret
  git-clang-format --style=file main
  git diff --exit-code
else
  enable_update_ret
  echo "'git-clang-format' not found. Skipping check"
fi

echo "***** tabs *****"
${scripts_dir}/check_tabs.sh

echo "***** yamllint *****"
disable_update_ret
if exists yamllint; then
  enable_update_ret
  ${scripts_dir}/run_yamllint.sh
else
  enable_update_ret
  echo "'yamllint' not found. Skipping check"
fi

echo "***** markdownlint *****"
# Don't fail script if condition is false
disable_update_ret
if exists markdownlint; then
  enable_update_ret
  ${scripts_dir}/run_markdownlint.sh
else
  enable_update_ret
  echo "'markdownlint' not found. Skipping check"
fi

echo "***** Path Lengths *****"
./build_tools/scripts/check_path_lengths.py

echo "***** Generates CMake files *****"
./build_tools/scripts/generate_cmake_files.sh
git add -A
git diff HEAD --exit-code

echo "***** Check BUILD files are not named BUILD (prefer BUILD.bazel) *****"
if [[ $(git ls-files '**/BUILD') ]]; then
  echo "failure: found files named BUILD. Please rename the following files to BUILD.bazel:"
  git ls-files '**/BUILD'
  (exit 1)
fi

if (( "${FINAL_RET}" != 0 )); then
  echo "Encountered failures. Check error messages and changes to the working" \
       "directory and git index (which may contain fixes) and try again."
fi

exit "${FINAL_RET}"
