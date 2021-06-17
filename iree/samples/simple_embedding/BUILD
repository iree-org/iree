# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT})
""",
    inline = True,
)

# TODO(llvm-bazel/#228): re-enable bazel.
#cc_binary(
#    name = "simple_embedding_dylib",
#    srcs = [
#        "device_dylib.c",
#        "simple_embedding.c",
#    ],
#    deps = [
#        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_riscv_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_x86_64_c",
#        "//iree/base",
#        "//iree/hal",
#        "//iree/hal/local",
#        "//iree/hal/local:task_driver",
#        "//iree/hal/local/loaders:embedded_library_loader",
#        "//iree/modules/hal",
#        "//iree/task:api",
#        "//iree/vm",
#        "//iree/vm:bytecode_module",
#    ],
#)
#
#cc_binary(
#    name = "simple_embedding_local_sync",
#    srcs = [
#        "device_local_sync.c",
#        "simple_embedding.c",
#    ],
#    deps = [
#        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_riscv_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_x86_64_c",
#        "//iree/base",
#        "//iree/hal",
#        "//iree/hal/local",
#        "//iree/hal/local:sync_driver",
#        "//iree/hal/local/loaders:embedded_library_loader",
#        "//iree/modules/hal",
#        "//iree/vm",
#        "//iree/vm:bytecode_module",
#    ],
#)
#
#iree_bytecode_module(
#    name = "simple_embedding_test_bytecode_module_dylib_x86_64",
#    src = "simple_embedding_test.mlir",
#    c_identifier = "iree_samples_simple_embedding_test_module_dylib_x86_64",
#    flags = [
#        "-iree-input-type=mhlo",
#        "-iree-mlir-to-vm-bytecode-module",
#        "-iree-hal-target-backends=dylib-llvm-aot",
#        "-iree-llvm-target-triple=x86_64-pc-linux-elf",
#        "-iree-llvm-link-embedded=true",
#        "-iree-llvm-debug-symbols=false",
#    ],
#)
#
#iree_bytecode_module(
#    name = "simple_embedding_test_bytecode_module_dylib_riscv_64",
#    src = "simple_embedding_test.mlir",
#    c_identifier = "iree_samples_simple_embedding_test_module_dylib_riscv_64",
#    flags = [
#        "-iree-input-type=mhlo",
#        "-iree-mlir-to-vm-bytecode-module",
#        "-iree-hal-target-backends=dylib-llvm-aot",
#        "-iree-llvm-target-triple=riscv64-pc-linux-elf",
#        "-iree-llvm-target-cpu=generic-rv64",
#        "-iree-llvm-target-cpu-features=+m,+a,+f,+d,+c",
#        "-iree-llvm-target-abi=lp64d",
#        "-iree-llvm-link-embedded=true",
#        "-iree-llvm-debug-symbols=false",
#    ],
#)
#
#iree_bytecode_module(
#    name = "simple_embedding_test_bytecode_module_dylib_arm_64",
#    src = "simple_embedding_test.mlir",
#    c_identifier = "iree_samples_simple_embedding_test_module_dylib_arm_64",
#    flags = [
#        "-iree-input-type=mhlo",
#        "-iree-mlir-to-vm-bytecode-module",
#        "-iree-hal-target-backends=dylib-llvm-aot",
#        "-iree-llvm-target-triple=aarch64-pc-linux-elf",
#        "-iree-llvm-link-embedded=true",
#        "-iree-llvm-debug-symbols=false",
#    ],
#)
#
#run_binary_test(
#    name = "simple_embedding_dylib_test",
#    test_binary = ":simple_embedding_dylib",
#)
#
#run_binary_test(
#    name = "simple_embedding_local_sync_test",
#    test_binary = ":simple_embedding_local_sync",
#)

iree_cmake_extra_content(
    content = """
endif()

if(${IREE_TARGET_BACKEND_VULKAN-SPIRV})
""",
    inline = True,
)

cc_binary(
    name = "simple_embedding_vulkan",
    srcs = [
        "device_vulkan.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_vulkan_c",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/vulkan/registration",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_vulkan",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_vulkan",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vulkan-spirv",
        "-iree-llvm-debug-symbols=false",
    ],
)

run_binary_test(
    name = "simple_embedding_vulkan_test",
    test_binary = ":simple_embedding_vulkan",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
