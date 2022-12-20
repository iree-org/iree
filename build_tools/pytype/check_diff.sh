#!/bin/bash
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Uses git diff to run pytype on changed files.
# Example Usage:
#   Defaults to comparing against 'main'.
#     ./build_tools/pytype/check_diff.sh
#   A specific branch can be specified.
#     ./build_tools/pytype/check_diff.sh  google
#   Or all python files outside of './third_party/' can be checked.
#     ./build_tools/pytype/check_diff.sh  all

DIFF_TARGET="${1:-main}"
echo "Running pycheck against '${DIFF_TARGET?}'"

if [[ "${DIFF_TARGET?}" = "all" ]]; then
  FILES=$(find -name "*\.py" -not -path "./third_party/*")
else
  FILES=$(git diff --diff-filter=d --name-only "${DIFF_TARGET?}" | grep '.*\.py$' | grep -vP 'lit.cfg.py')
fi


# We seperate the python files into multiple pytype calls because otherwise
# Ninja gets confused. See https://github.com/google/pytype/issues/198
BASE=$(echo "${FILES?}" | \
       grep -vP '^(\./)?integrations/*$' | \
       grep -vP '(\./)?setup\.py$')
IREE_TF=$(echo "${FILES?}" | \
          grep -P '^(\./)?integrations/tensorflow/bindings/python/iree/tf/.*')
IREE_XLA=$(echo "${FILES?}" | \
           grep -P '^(\./)?integrations/tensorflow/bindings/python/iree/xla/.*')
COMPILER=$(echo "${FILES?}" | \
           grep -P '^(\./)?integrations/tensorflow/compiler/.*')
E2E=$(echo "${FILES?}" | grep -P '^(\./)?integrations/tensorflow/e2e/.*')

function check_files() {
  # $1: previous return code
  # $2...: files to check
  if [[ -z "${@:2}" ]]; then
    echo "No files to check."
    echo
    return "${1?}"
  fi

  # We disable import-error and module-attr because pytype doesn't have access
  # to all dependencies and pyi-error because of the way the bindings imports
  # work.
  # xargs is set to high arg limits to avoid multiple pytype invocations and
  # will hard fail if the limits are exceeded.
  # See https://github.com/bazelbuild/bazel/issues/12479
  echo "${@:2}" | \
    xargs --max-args 1000000 --max-chars 1000000 --exit \
      python3 -m pytype --disable=import-error,pyi-error,module-attr -j $(nproc)
  EXIT_CODE="$?"
  echo
  if [[ "${EXIT_CODE?}" -gt "${1?}" ]]; then
    return "${EXIT_CODE?}"
  else
    return "${1?}"
  fi
}

MAX_CODE=0

echo "Checking .py files outside of integrations/"
check_files "${MAX_CODE?}" "${BASE?}"
MAX_CODE="$?"

echo "Checking .py files in integrations/tensorflow/bindings/python/iree/tf/.*"
check_files "${MAX_CODE?}" "${IREE_TF?}"
MAX_CODE="$?"

echo "Checking .py files in integrations/tensorflow/bindings/python/iree/xla/.*"
check_files "${MAX_CODE?}" "${IREE_XLA?}"
MAX_CODE="$?"

echo "Checking .py files in integrations/tensorflow/compiler/.*"
check_files "${MAX_CODE?}" "${COMPILER?}"
MAX_CODE="$?"

echo "Checking .py files in integrations/tensorflow/e2e/.*"
check_files "${MAX_CODE?}" "${E2E?}"
MAX_CODE="$?"


if [[ "${MAX_CODE?}" -ne "0" ]]; then
  echo "One or more pytype checks failed."
  echo "You can view these errors locally by running"
  echo "  ./build_tools/pytype/check_diff.sh ${DIFF_TARGET?}"
  exit "${MAX_CODE?}"
fi
