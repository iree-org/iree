// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/compiler/tool_entry_points_api.h"

int main(int argc, char **argv) {
  IREE_TRACE_APP_ENTER();
  int exit_code = ireeCompilerRunMain(argc, argv);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
