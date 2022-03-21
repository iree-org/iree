// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace linalg {

/// Find all operations in `containerOp` that are matched by the specified PDL
/// `matchOp`, which is located in the same parent ModuleOp as `matchOp`.
FailureOr<SmallVector<Operation *>> findMatchingOps(transform::MatchOp matchOp,
                                                    SymbolRefAttr pattern,
                                                    Operation *containerOp);

inline FailureOr<SmallVector<Operation *>>
findMatchingOps(transform::MatchOp matchOp, Operation *containerOp) {
  return findMatchingOps(matchOp, matchOp.targetMatcher(), containerOp);
}

} // namespace linalg
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H
