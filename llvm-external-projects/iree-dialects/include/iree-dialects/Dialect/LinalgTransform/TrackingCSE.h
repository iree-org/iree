//===-- TrackingCSE.h - Special common subexpr elimination ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H

namespace mlir {
class DominanceInfo;
struct LogicalResult;
class Operation;
struct RewriteListener;

LogicalResult
eliminateCommonSubexpressionsWithTrackedOps(Operation *root,
                                            RewriteListener &listener,
                                            DominanceInfo *domInfo = nullptr);
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
