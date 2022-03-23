// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_RTL_LINKAGE_ANALYSIS_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_RTL_LINKAGE_ANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;

namespace iree_compiler {
namespace IREE {
namespace PYDM {

/// An analysis of the external linkage which must be satisfied.
class LinkageAnalysis {
public:
  LinkageAnalysis(Operation *moduleOp);

  /// Whether there are any external functions that need resolution.
  bool hasExternFuncs() const { return !externFuncOps.empty(); }

  /// Gets a read-only list of the extern func ops.
  const llvm::SmallVector<Operation *> &getExternFuncOps() const {
    return externFuncOps;
  }

private:
  llvm::SmallVector<Operation *> externFuncOps;
};

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_RTL_LINKAGE_ANALYSIS_H
