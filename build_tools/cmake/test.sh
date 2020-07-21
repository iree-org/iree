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

# Run all(ish) IREE tests with CTest. Designed for CI, but can be run manually.
# Assumes that the project has already been built at ${REPO_ROOT}/build (e.g.
# with build_tools/cmake/clean_build.sh)

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

# Respect the user setting, but default to as many jobs as we have cores.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)}

# Respect the user setting, but default to turning off the vulkan tests
# and turning on the llvmjit ones.
export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-1}
export IREE_LLVMJIT_DISABLE=${IREE_LLVMJIT_DISABLE:-0}

# Tests to exclude by label. In addition to any custom labels (which are carried
# over from Bazel tags), every test should be labeled with the directory it is
# in.
declare -a label_exclude_args=(
  # Exclude specific labels.
  # Put the whole label with anchors for exact matches.
  # For example:
  #   ^nokokoro$
  ^nokokoro$

  # Exclude all tests in a directory.
  # Put the whole directory with anchors for exact matches.
  # For example:
  #   ^bindings/python/pyiree/rt$

  # Exclude all tests in some subdirectories.
  # Put the whole parent directory with only a starting anchor.
  # Use a trailing slash to avoid prefix collisions.
  # For example:
  #   ^bindings/
)

if [[ "${IREE_VULKAN_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^driver=vulkan$")
fi
if [[ "${IREE_LLVMJIT_DISABLE?}" == 1 ]]; then
  label_exclude_args+=("^driver=llvm$")
fi

# Join on "|"
label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]?}"))"

cd ${ROOT_DIR?}/build
ctest --output-on-failure --label-exclude "${label_exclude_regex?}"
