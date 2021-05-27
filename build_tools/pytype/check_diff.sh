#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
  FILES=$(git diff --diff-filter=d --name-only "${DIFF_TARGET?}" | grep '.*\.py')
fi


# We seperate the python files into multiple pytype calls because otherwise
# Ninja gets confused. See https://github.com/google/pytype/issues/198
BASE=$(echo "${FILES?}" | grep -vP '^(\./)?integrations/*')
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

  # We disable import-error because pytype doesn't have access to bazel.
  # We disable pyi-error because of the way the bindings imports work.
  echo "${@:2}" | \
    xargs python3 -m pytype --disable=import-error,pyi-error -j $(nproc)
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
