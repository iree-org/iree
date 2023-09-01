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

add_subdirectory(custom_dispatch)
add_subdirectory(custom_module)
add_subdirectory(dynamic_shapes)
add_subdirectory(emitc_modules)
add_subdirectory(py_custom_module)
add_subdirectory(simple_embedding)
add_subdirectory(static_library)
add_subdirectory(variables_and_state)
