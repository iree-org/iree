// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

struct IREEOneShotBufferizationOptions
    : public mlir::bufferization::OneShotBufferizationOptions {
  // TODO(#12933): Because of regressions in CUDA backend, there is an
  // option to keep a legacy mode of not representing the offset in the
  // type. Remove once the bug is fixed.
  bool embedSubspanOffsetIntoMemRefType = true;
};

// Register all interfaces needed for bufferization.
void registerBufferizationInterfaces(DialectRegistry &registry);

// Eliminate tensor.empty ops that are anchored on flow store ops.
LogicalResult storeTensorOpAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op,
    bufferization::OneShotAnalysisState &state);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_INTERFACES_BUFFERIZATIONINTERFACES_H_
