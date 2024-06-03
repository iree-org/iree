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

echo "***** pre-commit *****"
pre-commit run

echo "***** pytype *****"
./build_tools/pytype/check_diff.sh

if (( "${FINAL_RET}" != 0 )); then
  echo "Encountered failures when running: '${FAILING_CMD}'. Check error" \
       "messages and changes to the working directory and git index (which" \
       "may contain fixes) and try again."
fi

exit "${FINAL_RET}"
