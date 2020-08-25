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

# Checks for tabs in files modified vs the specified reference commit (default
# "main")

set -x
set -o pipefail

BASE_REF="${1:-main}"

declare -a excluded_files_patterns=(
  "^.gitmodules$"
  "/third_party/"
  "^third_party/"
)

# Join on |
excluded_files_pattern="$(IFS="|" ; echo "${excluded_files_patterns[*]?}")"

git diff --name-only "${BASE_REF?}" | \
  grep -v -E "${excluded_files_pattern?}" | \
  xargs \
    grep --with-filename --line-number --perl-regexp '\t'

if (( "$?" == 0 )); then
  echo "Changed files include tabs. Please use spaces.";
  exit 1;
fi
