// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_TILINGUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_TILINGUTILS_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

FailureOr<scf::SCFTileAndFuseResult>
tileConsumerAndFuseProducersWithReshapeFusion(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_TILINGUTILS_H_
