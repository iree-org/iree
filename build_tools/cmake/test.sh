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
    iree_test_e2e_xla_abs.mlir.test
    iree_test_e2e_xla_add.mlir.test
    iree_test_e2e_xla_batch_norm_inference.mlir.test
    iree_test_e2e_xla_compare.mlir.test
    iree_test_e2e_xla_constants.mlir.test
    iree_test_e2e_xla_conv.mlir.test
    iree_test_e2e_xla_cos.mlir.test
    iree_test_e2e_xla_dot.mlir.test
    iree_test_e2e_xla_exp.mlir.test
    iree_test_e2e_xla_fullyconnected.mlir.test
    iree_test_e2e_xla_gemm.mlir.test
    iree_test_e2e_xla_gemm_large.mlir.test
    iree_test_e2e_xla_log.mlir.test
    iree_test_e2e_xla_max_float.mlir.test
    iree_test_e2e_xla_max_int.mlir.test
    iree_test_e2e_xla_min_float.mlir.test
    iree_test_e2e_xla_min_int.mlir.test
    iree_test_e2e_xla_mnist.mlir.test
    iree_test_e2e_xla_multiple_return.mlir.test
    iree_test_e2e_xla_reduce_float.mlir.test
    iree_test_e2e_xla_reduce_int.mlir.test
    iree_test_e2e_xla_rem.mlir.test
    iree_test_e2e_xla_reshape.mlir.test
    iree_test_e2e_xla_reshape_adddims.mlir.test
    iree_test_e2e_xla_reshape_dropdims.mlir.test
    iree_test_e2e_xla_rsqrt.mlir.test
    iree_test_e2e_xla_scalar.mlir.test
    iree_test_e2e_xla_select.mlir.test
    iree_test_e2e_xla_sqrt.mlir.test
    iree_test_e2e_xla_through_std.mlir.test
    iree_test_e2e_xla_while.mlir.test
    iree_vm_bytecode_module_benchmark # Make this test not take an eternity
    bindings_python_pyiree_rt_function_abi_test
    bindings_python_pyiree_rt_system_api_test
    bindings_python_pyiree_rt_vm_test
    bindings_python_pyiree_rt_hal_test # TODO: Enable after the VM is fixed
    bindings_python_pyiree_compiler_compiler_test # TODO: Enable after the VM is fixed
)

# Join with | and add anchors
EXCLUDED_REGEX="^($(IFS="|" ; echo "${EXCLUDED_TESTS[*]?}"))$"

cd ${ROOT_DIR?}/build
ctest -E "${EXCLUDED_REGEX?}"

