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

set -x
set -e
set -o pipefail

GREEN_COMMIT="${1:-main}"
BRANCH="${BRANCH:-main-to-google}"

git sync
git checkout "${GREEN_COMMIT?}"
git submodule update
git checkout -B "${BRANCH?}"
git push -f fork "${BRANCH?}"
gh pr create \
    --base google \
    --title="Merge main -> google" \
    --body="$(git log google.. --decorate=no --pretty='format:* %h %<(80,trunc)%s')"
