// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Framework/WorkItem.h"

using namespace mlir;
using namespace mlir::iree_compiler;

WorkItem WorkItem::clone() { return WorkItem(root.clone()); }

int64_t WorkItem::getComplexityScore() {
  // TODO: Guide the reducer using this complexity score.
  return 0;
}
