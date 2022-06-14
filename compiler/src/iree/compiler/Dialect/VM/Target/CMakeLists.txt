# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_INIT_INTERNAL_DEPS)
add_subdirectory(Bytecode)
list(APPEND _INIT_INTERNAL_DEPS iree::compiler::Dialect::VM::Target::Bytecode)
if(IREE_OUTPUT_FORMAT_C)
  add_subdirectory(C)
  list(APPEND _INIT_INTERNAL_DEPS iree::compiler::Dialect::VM::Target::C)
endif()

iree_cc_library(
  NAME
    init_targets
  HDRS
    "init_targets.h"
  DEPS
    ${_INIT_INTERNAL_DEPS}
  PUBLIC
)
