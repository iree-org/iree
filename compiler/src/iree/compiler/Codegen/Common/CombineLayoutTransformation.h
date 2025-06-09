// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_
#define IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// Describes the configuration for distributing a fully parallel iteration
/// space. The rank of `tileSizes` and `mapping` should match the rank of the
/// iteration space for which the DistributionConfig is described.
struct DistributionConfig {
  /// The tile sizes for each worker of the distribution. The strides of the
  /// tiles are expected to be 1.
  SmallVector<int64_t> tileSizes;
  /// Distribution mapping for the distributed loops (e.g., the `mapping`
  /// attribute of an scf.forall op).
  SmallVector<Attribute> mapping;
};

/// Type for a callback that determines how to distribute the writing of pad
/// values to an output buffer. The function takes in the iteration bounds of
/// the original pad op, and returns a list of `DistributionConfig`s, each of
/// which describes a level of distribution. The first DistributionConfig in
/// the list represents the outermost distribution loop set.
using PadDistributionConfigFn = function_ref<SmallVector<DistributionConfig>(
    ArrayRef<int64_t> iterationBounds, MLIRContext *)>;

/// Combines any layout/indexing transformation ops at the ends of a dispatch.
/// Finds `iree_codegen.store_to_buffer` ops in the `funcOp`, and combines any
/// layout transformation ops (like expand_shape, transpose, pack, etc.) that
/// produce the tensor being stored into a single `iree_linalg_ext.map_scatter`
/// op.
///
/// This transformation will also combine `tensor.pad` ops into the map_scatter
/// op, by moving the writing of the padding values to after the store_to_buffer
/// op, and writing the padding values directly to the output buffer of the
/// store_to_buffer. The writes of the pad values will be distributed based on
/// the `DistributionConfig`s returned by `padDistributionConfigFn`, and then
/// the inner distributed tile will be tiled to a loop nest of memref.store ops.
LogicalResult
combineLayoutTransformation(MLIRContext *ctx, FunctionOpInterface funcOp,
                            PadDistributionConfigFn padDistributionConfigFn);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_
