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

export IREE_VULKAN_DISABLE=${IREE_VULKAN_DISABLE:-1}

EXCLUDED_TESTS=(
    iree_compiler_Translation_SPIRV_LinalgToSPIRV_test_pw_add.mlir.test
    iree_hal_vulkan_dynamic_symbols_test
    iree_modules_check_check_test
    iree_modules_tensorlist_tensorlist_test
    iree_tools_test_multiple_args.mlir.test
    iree_tools_test_scalars.mlir.test
    iree_tools_test_simple.mlir.test
    iree_tools_vm_util_test
    iree_vm_bytecode_module_benchmark # Make this test not take an eternity
)

# Join with | and add anchors
EXCLUDED_REGEX="^($(IFS="|" ; echo "${EXCLUDED_TESTS[*]?}"))$"

cd ${ROOT_DIR?}/build
ctest -E "${EXCLUDED_REGEX?}"

