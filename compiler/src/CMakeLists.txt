# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_setup_c_src_root(
  IMPLICIT_DEFS_TARGET iree_compiler_defs
  IMPLICIT_DEFS_INSTALL_COMPONENT IREERuntimeLibraries-Compiler
  IMPLICIT_DEFS_EXPORT_SET Compiler
)

add_subdirectory(iree/compiler)
