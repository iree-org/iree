# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    identifier = "iree_samples_simple_embedding_rv64_test_module",
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

cc_binary(
    name = "simple_embedding_dylib",
    srcs = [
        "device_dylib.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_c",
        ":simple_embedding_test_llvm_aot_rv64",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:task_driver",
        "//iree/hal/local/loaders:embedded_library_loader",
        "//iree/hal/local/loaders:legacy_library_loader",
        "//iree/modules/hal",
        "//iree/task:api",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

cc_binary(
    name = "simple_embedding_vulkan",
    srcs = [
        "device_vulkan.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_c",
        ":simple_embedding_test_llvm_aot_rv64",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/vulkan/registration",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

cc_binary(
    name = "simple_embedding_local_sync",
    srcs = [
        "device_local_sync.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_c",
        ":simple_embedding_test_llvm_aot_rv64",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:sync_driver",
        "//iree/hal/local/loaders:embedded_library_loader",
        "//iree/hal/local/loaders:legacy_library_loader",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-hal-target-backends=vulkan-spirv",
        "-iree-llvm-debug-symbols=false",
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
    test_binary = ":simple_embedding_dylib",
)

run_binary_test(
    name = "simple_embedding_local_sync_test",
    test_binary = ":simple_embedding_local_sync",
)

run_binary_test(
    name = "simple_embedding_vulkan_test",
    test_binary = ":simple_embedding_vulkan",
)
