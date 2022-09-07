// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace iree_compiler {

/// Populate patterns related to tile and distribute to workgroups.
void populateTileAndDistributeToWorkgroupsPatterns(
    RewritePatternSet &patterns, linalg::LinalgTilingOptions options,
    linalg::LinalgTransformationFilter filter, bool specialize);

/// Populate patterns that fold tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns);

}  // namespace iree_compiler
}  // namespace mlir
