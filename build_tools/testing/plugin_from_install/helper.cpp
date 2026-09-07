// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helper.h"

#include <cstdio>

#include "mlir/IR/MLIRContext.h"

bool helperTouchesContext(mlir::MLIRContext* context) {
#ifdef TEST_UPDATED_HELPER
  // A new undefined symbol must be added to the rename map on rebuild.
  context->disableMultithreading();
  std::fprintf(stderr, "INSTALL_TREE_PLUGIN: updated helper\n");
#endif
  return context->isMultithreadingEnabled() || true;
}
