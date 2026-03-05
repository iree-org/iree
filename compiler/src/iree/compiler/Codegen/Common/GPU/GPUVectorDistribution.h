// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_

#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"

namespace mlir::iree_compiler {

/// Distribute vector operations in the IR rooted at `root`.
///
/// The flow of distribution looks like:
///   - Make `options` set some initial information about how to distribute
///     some vector values. This is usually done on operations like
///     vector.contract, vector.transfer_read/vector.transfer_write,
///     vector.multi_reduction, where we are trying to target specific
///     hardware instructions. This information is provided in the form of a
///     layout for the value.
///   - Run a global analysis to determine how to distribute rest of the vector
///     values keeping the initial anchors in mind.
///   - Use the analysis information to distribute each operation.
LogicalResult
distributeVectorOps(Operation *root, RewritePatternSet &distributionPatterns,
                    IREE::VectorExt::VectorLayoutOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
