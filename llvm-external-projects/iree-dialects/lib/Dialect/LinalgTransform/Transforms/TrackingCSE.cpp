//===-- TrackingCSE.cpp - Common subexpr elimination keeping track of ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgTransform/TrackingCSE.h"

#include "Transforms/ListenerCSE.h"

using namespace mlir;

LogicalResult mlir::eliminateCommonSubexpressionsWithTrackedOps(
    Operation *root, RewriteListener &listener, DominanceInfo *domInfo) {
  return eliminateCommonSubexpressions(root, domInfo, &listener);
}
