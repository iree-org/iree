#!/bin/bash

# Copyright 2019 Google LLC
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
set -e

ls -lR ${RUNFILES_DIR}

# Detect whether cygwin/msys2 paths need to be translated.
cygpath="$(which cygpath 2>/dev/null)"

# Bazel helpfully puts all data deps in the ${RUNFILES_DIR}, but
# it unhelpfully preserves the nesting with no way to reason about
# it generically. run_lit expects that anything passed in the runfiles
# can be found on the path for execution. So we just iterate over the
# entries in the MANIFEST and extend the PATH.
SUBPATH=""
for runfile_path in $(find "${RUNFILES_DIR}" -executable -print); do
  # Prepend so that local things override.
  EXEDIR="$(dirname ${runfile_path})"
  if ! [ -z "$cygpath" ]; then
    EXEDIR="$($cygpath -u "$EXEDIR")"
  fi
  SUBPATH="${EXEDIR}:$SUBPATH"
done

echo "run_lit.sh: $1"
echo "PWD=$(pwd)"

# Extract the test first line and assume it starts with:
# // RUN: ...
firstline="$(egrep "^// RUN: " "$1" | head -n 1)"
if [ -z "${firstline}" ]; then
  echo "ERROR: Could not find lit '// RUN: ' line"
  exit 1
fi

echo "FIRSTLINE: $firstline"
match="${firstline%%// RUN: *}"
command="${firstline##// RUN: }"
echo "RUN MATCH: $match"
echo "COMMAND: $command"

if ! [ -z "$match" ]; then
  echo "Test file does not start with '// RUN: ': '$firstline'"
  exit 2
fi

# Substitute any embedded '%s' with the file name.
full_command="${command//\%s/$1}"
echo "FULL COMMAND: $full_command"

# Run it.
export PATH="$SUBPATH"
echo "PATH=$PATH"
echo "RUNNING TEST:"
echo "----------------"
set -x
eval "$full_command"
echo "--- COMPLETE ---"
exit $?
