//===-- PDL.h - Interoperability with PDL ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace linalg {

/// Find all operations in `module` that are matched by the specified PDL
/// pattern, which is also located in `module`.
FailureOr<SmallVector<Operation *>> findMatchingOps(Operation *op,
                                                    SymbolRefAttr pattern,
                                                    ModuleOp module);
inline FailureOr<SmallVector<Operation *>> findMatchingOps(
    transform::MatchOp op, ModuleOp module) {
  return findMatchingOps(op, op.targetMatcher(), module);
}

}  // namespace linalg
}  // namespace mlir

#endif  // IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_PDL_H
