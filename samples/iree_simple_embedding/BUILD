# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:native_binary.bzl", "native_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if((IREE_HAL_DRIVER_VMVX OR IREE_HAL_DRIVER_VMVX_SYNC) AND
   (IREE_TARGET_BACKEND_VMVX OR DEFINED IREE_HOST_BINARY_ROOT))
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
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:sync_driver",
        "//runtime/src/iree/hal/local/loaders:vmvx_module_loader",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
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
    deps = [
        "//samples:defs",
    ],
)

native_test(
    name = "simple_embedding_vmvx_sync_test",
    src = ":simple_embedding_vmvx_sync",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_cmake_extra_content(
    content = """
if((IREE_HAL_DRIVER_DYLIB OR IREE_HAL_DRIVER_DYLIB_SYNC) AND
   (IREE_TARGET_BACKEND_DYLIB_LLVM_AOT OR DEFINED IREE_HOST_BINARY_ROOT))
""",
    inline = True,
)

cc_binary(
    name = "simple_embedding_embedded_sync",
    srcs = [
        "device_embedded_sync.c",
        "simple_embedding.c",
    ],
    deps = [
        ":simple_embedding_test_bytecode_module_dylib_arm_32_c",
        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
        ":simple_embedding_test_bytecode_module_dylib_riscv_32_c",
        ":simple_embedding_test_bytecode_module_dylib_riscv_64_c",
        ":simple_embedding_test_bytecode_module_dylib_x86_64_c",
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:sync_driver",
        "//runtime/src/iree/hal/local/loaders:embedded_library_loader",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_dylib_x86_64",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_dylib_x86_64",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-llvm-target-triple=x86_64-pc-linux-elf",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
    ],
    deps = [
        "//samples:defs",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_dylib_riscv_32",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_dylib_riscv_32",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-llvm-target-triple=riscv32-pc-linux-elf",
        "-iree-llvm-target-cpu=generic-rv32",
        "-iree-llvm-target-cpu-features=+m,+f",
        "-iree-llvm-target-abi=ilp32",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
    ],
    deps = [
        "//samples:defs",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_dylib_riscv_64",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_dylib_riscv_64",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-llvm-target-triple=riscv64-pc-linux-elf",
        "-iree-llvm-target-cpu=generic-rv64",
        "-iree-llvm-target-cpu-features=+m,+a,+f,+d,+c",
        "-iree-llvm-target-abi=lp64d",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
    ],
    deps = [
        "//samples:defs",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_dylib_arm_32",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_dylib_arm_32",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-llvm-target-triple=armv7a-pc-linux-elf",
        "-iree-llvm-target-float-abi=hard",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
    ],
    deps = [
        "//samples:defs",
    ],
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module_dylib_arm_64",
    src = "simple_embedding_test.mlir",
    c_identifier = "iree_samples_simple_embedding_test_module_dylib_arm_64",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-llvm-target-triple=aarch64-pc-linux-elf",
        "-iree-llvm-debug-symbols=false",
        "-iree-vm-bytecode-module-strip-source-map=true",
        "-iree-vm-emit-polyglot-zip=false",
    ],
    deps = [
        "//samples:defs",
    ],
)

native_test(
    name = "simple_embedding_embedded_sync_test",
    src = ":simple_embedding_embedded_sync",
)

iree_cmake_extra_content(
    content = """
if(IREE_HAL_DRIVER_DYLIB)
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
        ":simple_embedding_test_bytecode_module_dylib_arm_64_c",
        ":simple_embedding_test_bytecode_module_dylib_riscv_64_c",
        ":simple_embedding_test_bytecode_module_dylib_x86_64_c",
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:task_driver",
        "//runtime/src/iree/hal/local/loaders:embedded_library_loader",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/task:api",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
    ],
)

native_test(
    name = "simple_embedding_dylib_test",
    src = ":simple_embedding_dylib",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_cmake_extra_content(
    content = """
endif()

if(IREE_HAL_DRIVER_VULKAN AND
   (IREE_TARGET_BACKEND_VULKAN_SPIRV OR DEFINED IREE_HOST_BINARY_ROOT))
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
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/vulkan/registration",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
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
    deps = [
        "//samples:defs",
    ],
)

native_test(
    name = "simple_embedding_vulkan_test",
    src = ":simple_embedding_vulkan",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

# Disabled because CUDA is not universally available and Bazel does not
# support configurability in a reasonable or useful way.
# iree_cmake_extra_content(
#     content = """
# if(IREE_HAL_DRIVER_CUDA AND
#    (IREE_TARGET_BACKEND_CUDA OR DEFINED IREE_HOST_BINARY_ROOT))
# """,
#     inline = True,
# )

# cc_binary(
#     name = "simple_embedding_cuda",
#     srcs = [
#         "device_cuda.c",
#         "simple_embedding.c",
#     ],
#     deps = [
#         ":simple_embedding_test_bytecode_module_cuda_c",
#         "//runtime/src/iree/base",
#         "//runtime/src/iree/hal",
#         "//runtime/src/iree/hal/cuda/registration",
#         "//runtime/src/iree/modules/hal",
#         "//runtime/src/iree/vm",
#         "//runtime/src/iree/vm:bytecode_module",
#     ],
# )

# iree_bytecode_module(
#     name = "simple_embedding_test_bytecode_module_cuda",
#     src = "simple_embedding_test.mlir",
#     c_identifier = "iree_samples_simple_embedding_test_module_cuda",
#     flags = [
#         "-iree-input-type=mhlo",
#         "-iree-mlir-to-vm-bytecode-module",
#         "-iree-hal-target-backends=cuda",
#         "-iree-llvm-debug-symbols=false",
#     ],
# )

# # native_test(
# #     name = "simple_embedding_cuda_test",
# #     src = ":simple_embedding_cuda",
# #     tags = [
# #         "driver=cuda",
# #     ],
# # )

# iree_cmake_extra_content(
#     content = """
# iree_native_test(
#   NAME
#     "simple_embedding_cuda_test"
#   LABELS
#     "driver=cuda"
#   SRC
#     ::simple_embedding_cuda
# )

# endif()
# """,
#     inline = True,
# )
