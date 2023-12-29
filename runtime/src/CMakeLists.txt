# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Instruct install support that headers are installable from this root
# directory.
set(IREE_HDRS_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Otherwise undeclared libraries are considered part of the public 
# runtime libraries.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEDevLibraries-Runtime)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET "Runtime")

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "")

add_subdirectory(iree)
