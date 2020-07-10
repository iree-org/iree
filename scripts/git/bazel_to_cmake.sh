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


set -e
set -x
set -o pipefail

BASE="${1:-google}"
BRANCH="bazel-to-cmake-fix"

git update "${BASE?}"
git checkout -B "${BRANCH?}"
./build_tools/bazel_to_cmake/bazel_to_cmake.py

TITLE="Run Bazel -> CMake"
BODY='Updates CMake files to match Bazel BUILD

`./build_tools/bazel_to_cmake/bazel_to_cmake.py`
'

git commit -am "${TITLE?}"
git push -f fork "${BRANCH?}"
gh pr create --title="${TITLE?}" --body="${BODY?}" --base="${BASE?}"
