# Copyright 2019 Google LLC
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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")
load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# The prebuilt simple_embedding_test-llvm-aot_rv64 bytecode module is build with
# RISC-V 64-bit cross-compile toolchain (with RISCV_TOOLCHAIN_ROOT defined):
# iree-translate \
#     -iree-mlir-to-vm-bytecode-module \
#     -iree-hal-target-backends=dylib-llvm-aot \
#     -iree-llvm-target-triple=riscv64 \
#     -iree-llvm-target-cpu=generic-rv64 \
#     -iree-llvm-target-cpu-features="+m,+a,+f,+d,+c" \
#     -iree-llvm-target-abi=lp64d \
#     iree/samples/simple_embedding/simple_embedding_test.mlir \
#     -o /tmp/simple_embedding_test-llvm-aot_rv64.vmfb
c_embed_data(
    name = "simple_embedding_test_llvm_aot_rv64",
    srcs = [
        "data/simple_embedding_test-llvm-aot_rv64.vmfb",
    ],
    c_file_output = "simple_embedding_test_llvm_aot_rv64.c",
    flatten = True,
    h_file_output = "simple_embedding_test_llvm_aot_rv64.h",
)

cc_binary(
    name = "simple_embedding_run",
    srcs = ["simple_embedding_run.c"],
    deps = [
        ":simple_embedding_test_bytecode_module_c",
        ":simple_embedding_test_llvm_aot_rv64",
        "//iree/base:api",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

cc_binary(
    name = "simple_embedding_run_local_sync",
    srcs = ["simple_embedding_run_local_sync.c"],
    deps = [
        ":simple_embedding_test_bytecode_module_c",
        ":simple_embedding_test_llvm_aot_rv64",
        "//iree/base:api",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:sync_driver",
        "//iree/hal/local/loaders:legacy_library_loader",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT} OR NOT ${IREE_TARGET_BACKEND_VULKAN-SPIRV}
   OR NOT ${IREE_HAL_DRIVER_DYLIB} OR NOT ${IREE_HAL_DRIVER_VULKAN})
  return()
endif()
""",
    inline = True,
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module",
    src = "simple_embedding_test.mlir",
    c_output = True,
    cc_namespace = "iree::samples",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-hal-target-backends=vulkan-spirv",
    ],
)

# Bytecode module is built without cross-compilation. Bypass the following test if it
# is not built for the host machine.

iree_cmake_extra_content(
    content = """
# Bytecode is built without cross-compilation. Bypass the following test if it
# is not built for the host machine.

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|AMD64")
  return()
endif()
""",
    inline = True,
)

run_binary_test(
    name = "simple_embedding_dylib_test",
    args = ["dylib"],
    test_binary = ":simple_embedding_run",
)

run_binary_test(
    name = "simple_embedding_run_local_sync_test",
    test_binary = ":simple_embedding_run_local_sync",
)

run_binary_test(
    name = "simple_embedding_vulkan_test",
    args = ["vulkan"],
    test_binary = ":simple_embedding_run",
)
