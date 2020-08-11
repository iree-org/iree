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

# Creates a commit summary of commits between the specified refs.
#
# Get changes between the current HEAD commit and the latest fetch of the
# upstream repository main branch:
#   summarize_changes.sh upstream/main
#   summarize_changes.sh upstream/main HEAD
#
# Summarize commits between the local main-to-google branch and the latest fetch
# of the upstream repository main branch
#   summarize_changes.sh upstream/main main-to-google

set -e
set -o pipefail

BASE_REF="${1}"
NEW_REF="${2:-HEAD}"

# Print commits with their short hash and the first 80 characters of their
# commit title. Use awk to trim the trailing whitespace introduced by git log

git log \
  "${BASE_REF?}..${NEW_REF?}" \
  --decorate=no \
  --pretty='format:* %h %<(80,trunc)%s' \
  | awk '{$1=$1;print}'
