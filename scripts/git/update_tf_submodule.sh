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

TENSORFLOW_COMMIT="${1:-REMOTE}"
BRANCH="submodule-update"

git update google
git checkout -B "${BRANCH?}"

CMD="./scripts/git/update_tf_llvm_submodules.py --llvm_commit=KEEP --update_build_files=true --tensorflow_commit=${TENSORFLOW_COMMIT?}"

bash -c "${CMD?}"

TF_SHA="$(git submodule status third_party/tensorflow | awk '{print $1}' | cut -c -12)"

TF_COMMIT_NICKNAME=""
if [[ "${TENSORFLOW_COMMIT?}" == "REMOTE" ]]; then
  TF_COMMIT_NICKNAME=" current HEAD"
fi

TITLE="Integrate TF at ${TF_SHA?}"
BODY="$(cat <<-EOF
Updates TF to${TF_COMMIT_NICKNAME?}
(https://github.com/tensorflow/tensorflow/commit/${TF_SHA?})
and copies over the LLVM BUILD files.

\`${CMD?}\`
EOF
)"

git commit -am "${TITLE?}"
git push -f fork "${BRANCH?}"
gh pr create --title="${TITLE?}" --body="${BODY?}" --base="google"
