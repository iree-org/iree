# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

FetchContent_Declare(
  spirv-tools
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
  GIT_TAG ff07cfd86fa229525659f6b81058b3171a67bef1 # 2021-12-10
)

set(SKIP_SPIRV_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(spirv-tools)
FetchContent_GetProperties(spirv-tools SOURCE_DIR SPIRV_TOOLS_SOURCE)
