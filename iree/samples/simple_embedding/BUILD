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
if(${IREE_HAL_DRIVER_VMVX} AND (${IREE_TARGET_BACKEND_VMVX} OR DEFINED IREE_HOST_BINARY_ROOT))
""",
    inline = True,
)

cc_binary(
    name = "simple_embedding_vmvx_sync",
    srcs = [
        "device_vmvx_sync.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_vmvx_c",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:sync_driver",
        "//iree/hal/local/loaders:vmvx_module_loader",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_vmvx",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_vmvx",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmvx",
    ],
)

run_binary_test(
    name = "simple_embedding_vmvx_sync_test",
    test_binary = ":simple_embedding_vmvx_sync",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_cmake_extra_content(
    content = """
if(${IREE_HAL_DRIVER_DYLIB} AND (${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT} OR DEFINED IREE_HOST_BINARY_ROOT))
""",
    inline = True,
)

# TODO(llvm-bazel/#228): re-enable bazel.
#cc_binary(
#    name = "simple_embedding_embedded_sync",
#    srcs = [
#        "device_embedded_sync.c",
#        "simple_embedding.c",
#    ],
#    deps = [
#        ":simple_embedding_test_bytecode_module_dylib_arm_32_c",
#        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_riscv_32_c",
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
# TODO(#6121): Adjust the flags to resolve segfault if necessary.
#iree_bytecode_module(
#    name = "simple_embedding_test_bytecode_module_dylib_riscv_32",
#    src = "simple_embedding_test.mlir",
#    c_identifier = "iree_samples_simple_embedding_test_module_dylib_riscv_32",
#    flags = [
#        "-iree-input-type=mhlo",
#        "-iree-mlir-to-vm-bytecode-module",
#        "-iree-hal-target-backends=dylib-llvm-aot",
#        "-iree-llvm-target-triple=riscv32-pc-linux-elf",
#        "-iree-llvm-target-cpu=generic-rv32",
#        "-iree-llvm-target-cpu-features=+m,+f",
#        "-iree-llvm-target-abi=ilp32",
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
#    name = "simple_embedding_test_bytecode_module_dylib_arm_32",
#    src = "simple_embedding_test.mlir",
#    c_identifier = "iree_samples_simple_embedding_test_module_dylib_arm_32",
#    flags = [
#        "-iree-input-type=mhlo",
#        "-iree-mlir-to-vm-bytecode-module",
#        "-iree-hal-target-backends=dylib-llvm-aot",
#        "-iree-llvm-target-triple=armv7a-pc-linux-elf",
#        "-iree-llvm-target-float-abi=hard",
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
#    name = "simple_embedding_embedded_sync_test",
#    test_binary = ":simple_embedding_embedded_sync",
#)

iree_cmake_extra_content(
    content = """
if(${IREE_ENABLE_THREADING})
""",
    inline = True,
)

#cc_binary(
#    name = "simple_embedding_dylib",
#    srcs = [
#        "device_dylib.c",
#        "simple_embedding.c",
#    ],
#    deps = [
#        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
#        ":simple_embedding_test_bytecode_module_dylib_riscv_32_c",
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
#run_binary_test(
#    name = "simple_embedding_dylib_test",
#    test_binary = ":simple_embedding_dylib",
#)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_cmake_extra_content(
    content = """
endif()

if(${IREE_HAL_DRIVER_VULKAN} AND (${IREE_TARGET_BACKEND_VULKAN-SPIRV} OR DEFINED IREE_HOST_BINARY_ROOT))
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

if(${IREE_HAL_DRIVER_CUDA} AND (${IREE_TARGET_BACKEND_CUDA} OR DEFINED IREE_HOST_BINARY_ROOT))
""",
    inline = True,
)

cc_binary(
    name = "simple_embedding_cuda",
    srcs = [
        "device_cuda.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_cuda_c",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/cuda/registration",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_cuda",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_cuda",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=cuda",
        "-iree-llvm-debug-symbols=false",
    ],
)

run_binary_test(
    name = "simple_embedding_cuda_test",
    tags = [
        "uses_cuda_runtime",
    ],
    test_binary = ":simple_embedding_cuda",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
