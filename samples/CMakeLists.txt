# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Not CMake directories:
#   colab
#   models
#   vision_inference
# Samples that are actually external projects do not get directly
# included:
#   compiler_plugins

# Not in the default build: needs a riscv-none-elf GCC toolchain with
# semihosting support and an explicit --target samples/baremetal_riscv64/all.
add_subdirectory(baremetal_riscv64 EXCLUDE_FROM_ALL)
add_subdirectory(custom_dispatch)
add_subdirectory(custom_module)
add_subdirectory(dynamic_shapes)
add_subdirectory(emitc_modules)
add_subdirectory(external_transients)
add_subdirectory(hal)
add_subdirectory(multiple_modules)
add_subdirectory(py_custom_module)
add_subdirectory(simple_embedding)
add_subdirectory(sink_callback)
add_subdirectory(static_library)
add_subdirectory(transform_dialect)
add_subdirectory(variables_and_state)
add_subdirectory(webgpu)
