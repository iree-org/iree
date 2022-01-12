#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -e
set -o pipefail

EXPLICIT_PATH=""

# First argument is the src file. Remaining arguments are tools that should
# be on the path.
src_file="$1"
shift
for tool_exe in "$@"
do
  EXEDIR="$(dirname $tool_exe)"
  if ! [ -z "$cygpath" ]; then
    EXEDIR="$($cygpath -u "$EXEDIR")"
  fi
  EXPLICIT_PATH="${EXEDIR}:${EXPLICIT_PATH}"
done

echo "run_lit.sh: $src_file"
echo "PWD=$(pwd)"
echo "EXPLICIT_PATH=$EXPLICIT_PATH"

# For each "// RUN:" line, run the command.
runline_matches="$(egrep "^// RUN: " "$src_file")"
if [ -z "$runline_matches" ]; then
  echo "!!! No RUN lines found in test"
  exit 1
fi

echo "$runline_matches" | while read -r runline
do
  echo "RUNLINE: $runline"
  match="${runline%%// RUN: *}"
  command="${runline##// RUN: }"
  if [ -z "${command}" ]; then
    echo "ERROR: Could not extract command from runline"
    exit 1
  fi

  # Substitute any embedded '%s' with the file name.
  full_command="${command//\%s/$src_file}"

  # Run it.
  export PATH="$EXPLICIT_PATH:$PATH"
  echo "RUNNING TEST: $full_command"
  echo "----------------"
  if eval "$full_command"; then
    echo "--- COMPLETE ---"
  else
    echo "!!! ERROR EVALUATING: $full_command"
    exit 1
  fi
done
