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

#### TODO: Remove all of this implicit path/runfiles stuff. The CMake rules
#### pass explicit tools on the command line.
if [ -z "${RUNFILES_DIR}" ]; then
  if [ -f "CMakeCache.txt" ]; then
    # If running under CMake/CTest in the build directory, just scope to the
    # iree directory to avoid blowing up the search through things like
    # bazel out directories and the like.
    RUNFILES_DIR="$PWD/iree"
  else
    # Some versions of bazel do not set RUNFILES_DIR. Instead they just cd
    # into the directory.
    RUNFILES_DIR="$PWD"
  fi
fi

# Detect whether cygwin/msys2 paths need to be translated.
set +e  # Ignore errors if not found.
cygpath="$(which cygpath 2>/dev/null)"
set -e

function find_executables() {
  set -e
  local p="$1"
  if [ "$(uname)" == "Darwin" ]; then
    # For macOS, xtype isn't avaliable and perm can't use `/u=x,g=x,o=x` syntax.
    find "${p}" -type l -perm +111
  elif [ -z "$cygpath" ]; then
    # For non-windows, use the perm based executable check, which has been
    # supported by find for a very long time.
    find "${p}" -xtype f -perm /u=x,g=x,o=x -print
  else
    # For windows, always use the newer -executable find predicate (which is
    # not supported by ancient versions of find).
    find "${p}" -xtype f -executable -print
  fi
}

# Bazel helpfully puts all data deps in the ${RUNFILES_DIR}, but
# it unhelpfully preserves the nesting with no way to reason about
# it generically. run_lit expects that anything passed in the runfiles
# can be found on the path for execution. So we just iterate over the
# entries in the MANIFEST and extend the PATH.
for runfile_path in $(find_executables "${RUNFILES_DIR}"); do
  # Prepend so that local things override.
  EXEDIR="$(dirname ${runfile_path})"
  if ! [ -z "$cygpath" ]; then
    EXEDIR="$($cygpath -u "$EXEDIR")"
  fi
  IMPLICIT_PATH="${EXEDIR}:$IMPLICIT_PATH"
done
#### END OF DEPRECATED IMPLICIT PATH DISCOVERY

echo "run_lit.sh: $src_file"
echo "PWD=$(pwd)"
echo "EXPLICIT_PATH=$EXPLICIT_PATH"
echo "IMPLICIT_PATH=$IMPLICIT_PATH"

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
  export PATH="$EXPLICIT_PATH:$IMPLICIT_PATH:$PATH"
  echo "RUNNING TEST: $full_command"
  echo "----------------"
  if eval "$full_command"; then
    echo "--- COMPLETE ---"
  else
    echo "!!! ERROR EVALUATING: $full_command"
    exit 1
  fi
done
