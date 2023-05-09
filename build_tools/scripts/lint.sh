#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs all the lint checks that we run on GitHub locally. Will fail if any tool
# is missing.

# Keep this in sync with .github/workflows/lint.yml

# WARNING: this script *makes changes* to the working directory and the index.

set -uo pipefail

FINAL_RET=0
PREV_CMD=""

scripts_dir="$(dirname $0)"

function update_ret() {
  local prev_ret="$?"
  local cur_cmd="${BASH_COMMAND}"
  if [[ "${prev_ret}" -gt "${FINAL_RET}" ]]; then
    FINAL_RET="${prev_ret}"
    FAILING_CMD="${PREV_CMD}"
  fi
  PREV_CMD="${cur_cmd}"
}

# Analyze the exit code of the previous command before every command
trap update_ret DEBUG

echo "***** Uncommitted changes *****"
git add -A
git diff HEAD --exit-code

if [[ $? -ne 0 ]]; then
  echo "Found uncomitted changes in working directory. This script requires" \
        "all changes to be committed. All changes have been added to the" \
        "index. Please commit or clean all changes and try again."
  exit 1
fi

echo "***** Bazel -> CMake *****"
./build_tools/bazel_to_cmake/bazel_to_cmake.py
./build_tools/bazel_to_cmake/bazel_to_cmake.py --root_dir=integrations/tensorflow/e2e
git add -A
git diff HEAD --exit-code

echo "***** buildifier *****"
${scripts_dir}/run_buildifier.sh
git diff --exit-code

echo "***** yapf *****"
# Don't fail script if condition is false
git diff -U0 main | ./third_party/format_diff/format_diff.py yapf -i

echo "***** pytype *****"
./build_tools/pytype/check_diff.sh

echo "***** clang-format *****"
git-clang-format --style=file main
git diff --exit-code

echo "***** tabs *****"
"${scripts_dir}/check_tabs.sh"

echo "***** yamllint *****"
"${scripts_dir}/run_yamllint.sh"

echo "***** markdownlint *****"
"${scripts_dir}/run_markdownlint.sh"

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
  echo "Encountered failures when running: '${FAILING_CMD}'. Check error" \
       "messages and changes to the working directory and git index (which" \
       "may contain fixes) and try again."
fi

exit "${FINAL_RET}"
