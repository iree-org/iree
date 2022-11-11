// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace bufferization {
struct OneShotBufferizationOptions;
}  // namespace bufferization

namespace iree_compiler {

/// Eliminate tensor.empty ops to avoid buffer allocations.
LogicalResult eliminateEmptyTensors(
    Operation *op, const bufferization::OneShotBufferizationOptions &options);

/// Bufferize the given op with One-Shot Bufferize.
LogicalResult runIREEOneShotBufferize(
    Operation *op, const bufferization::OneShotBufferizationOptions &options);

/// Populate patterns related to tile and distribute to workgroups.
void populateTileAndDistributeToWorkgroupsPatterns(
    RewritePatternSet &patterns, linalg::LinalgTilingOptions options,
    IREE::LinalgExt::LinalgTransformationFilter filter);

/// Populate patterns that fold tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns);

}  // namespace iree_compiler
}  // namespace mlir
