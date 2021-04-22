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

set -o pipefail

BASE_REF="${1:-main}"

declare -a excluded_files_patterns=(
  "^\.gitmodules$"
  "/third_party/"
  "^third_party/"
  "\.pb$"
  "\.fb$"
  "\.jar$"
  "\.so$"
)

# Join on |
excluded_files_pattern="$(IFS="|" ; echo "${excluded_files_patterns[*]?}")"

readarray -t files < <(\
  (git diff --name-only --diff-filter=d "${BASE_REF}" || kill $$) \
    | grep -v -E "${excluded_files_pattern?}")

diff="$(grep --with-filename --line-number --perl-regexp '\t' "${files[@]}")"

grep_exit="$?"

if (( "${grep_exit?}" >= 2 )); then
  exit "${grep_exit?}"
fi

if (( "${grep_exit?}" == 0 )); then
  echo "Changed files include tabs. Please use spaces.";
  echo "$diff"
  exit 1;
fi
