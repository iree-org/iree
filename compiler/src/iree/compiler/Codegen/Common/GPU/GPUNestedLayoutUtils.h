// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_GPUNESTEDLAYOUTUTILS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_GPUNESTEDLAYOUTUTILS_H_

#include <optional>

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Given a set of base transfer |indices|, |offsets| for the batch/outer
/// dimensions, and distributed warp and thread indices, computes the indices
/// of the distributed transfer operation based on the |vectorLayout|.
SmallVector<Value> getTransferIndicesFromNestedLayout(
    OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
    IREE::VectorExt::NestedLayoutAttr vectorLayout, AffineMap permutationMap,
    ValueRange warpIndices, ValueRange threadIndices);

/// Computes the warp and thread indices for the given vector layout from a
/// single linearized thread ID.
LogicalResult populateWarpAndThreadIndices(
    RewriterBase &rewriter, Value threadId, int64_t subgroupSize,
    IREE::VectorExt::NestedLayoutAttr vectorLayout,
    SmallVector<Value> &warpIndices, SmallVector<Value> &threadIndices);

/// Returns the distributed shape with batch/outer dims set to 1 for
/// element-tile-granularity iteration.
SmallVector<int64_t>
getElementVectorTileShape(IREE::VectorExt::NestedLayoutAttr vectorLayout);

/// Builds the nested layout used by derived_thread_config from explicit
/// element tile sizes.
FailureOr<IREE::VectorExt::NestedLayoutAttr>
getDerivedThreadLayout(MLIRContext *context, ArrayRef<int64_t> workgroupSize,
                       ArrayRef<int64_t> logicalShape,
                       ArrayRef<int64_t> elementTile);

/// Computes the nested layout used for global-load DMA lowering. Returns
/// failure if no DMA size can evenly distribute the transfer across the
/// subgroup/thread layout.
FailureOr<IREE::VectorExt::NestedLayoutAttr> getGlobalLoadDMALayout(
    MLIRContext *context, ArrayRef<int64_t> shape, int64_t numThreads,
    int64_t subgroupSize, int64_t elementBitWidth, ArrayRef<int64_t> dmaSizes,
    std::optional<int64_t> swizzleAccessElems = std::nullopt);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUNESTEDLAYOUTUTILS_H_
