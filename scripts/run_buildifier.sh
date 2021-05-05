#!/bin/bash

# Copyright 2021 Google LLC
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

# Checks buildifier formatting and lint in files modified vs the specified
# reference commit (default "main")
# See https://github.com/bazelbuild/buildtools/tree/master/buildifier

set -o pipefail

BASE_REF="${1:-main}"

declare -a included_files_patterns=(
  "/BUILD$"
  "\.bazel$"
  "/WORKSPACE$"
  "\.bzl$"
)

declare -a excluded_files_patterns=(
  "/third_party/"
  "^third_party/"
)

# Join on |
included_files_pattern="$(IFS="|" ; echo "${included_files_patterns[*]?}")"
excluded_files_pattern="$(IFS="|" ; echo "${excluded_files_patterns[*]?}")"

readarray -t files < <(\
  (git diff --name-only --diff-filter=d "${BASE_REF}" || kill $$) \
    | grep -E "${included_files_pattern?}" \
    | grep -v -E "${excluded_files_pattern?}")

if [ ${#files[@]} -eq 0 ]; then
  echo "No Bazel files changed"
  exit 0
fi

echo "Fixing formatting with Buildifier"
buildifier "${files[@]}"

echo "Fixing automatically fixable lint with Buildifier"
buildifier -lint=fix "${files[@]}"

echo "Checking complicated lint with Buildifier"
# Sometimes people don't write docstrings and I am not going to make that
# blocking.
buildifier -lint=warn -warnings=-module-docstring,-function-docstring "${files[@]}"
