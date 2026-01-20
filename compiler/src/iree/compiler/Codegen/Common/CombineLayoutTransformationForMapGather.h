// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATIONFORMAPGATHER_H_
#define IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATIONFORMAPGATHER_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// Control function type for layout transformation combination into map_gather.
/// The control function takes a leaf of a relayout op chain, and returns a bool
/// indicating whether to combine the relayout op chain, starting from the leaf.
using CombineRelayoutOpsForGatherControlFnRef =
    function_ref<bool(OpResult leaf)>;
using CombineRelayoutOpsForGatherControlFn = std::function<bool(OpResult leaf)>;

namespace IREE::Codegen {
/// Enum defining the scope of the CombineLayoutTransformationForMapGatherPass.
///  - The `Dispatch` scope will combine layout transformation chains that
///    start from an `iree_codegen.load_from_buffer` op.
enum class GatherRelayoutCombinationScope { Dispatch };
} // namespace IREE::Codegen

/// Get the corresponding control function for the given scope. The control
/// function verifies that the producer chain starts from LoadFromBufferOp.
CombineRelayoutOpsForGatherControlFn getCombineRelayoutOpsForGatherControlFn(
    IREE::Codegen::GatherRelayoutCombinationScope scope);

/// Returns true if the `op` type has a folding pattern into
/// iree_linalg_ext.map_gather. Note: tensor.pad is NOT supported for
/// map_gather because there is no output memref to write padding values to.
bool isSupportedGatherRelayoutOp(Operation *op);

/// Fold the `op` into the `mapGatherOp` and return the resulting map_gather,
/// or failure if the transformation is not supported. The `op` should be a
/// supported relayout op that produces the source of the map_gather.
FailureOr<IREE::LinalgExt::MapGatherOp>
foldIntoMapGather(RewriterBase &rewriter, Operation *op,
                  IREE::LinalgExt::MapGatherOp mapGatherOp);

/// Combines any layout/indexing transformation ops at the starts of a dispatch.
/// Finds `iree_codegen.load_from_buffer` ops in the `funcOp`, and combines any
/// layout transformation ops (like expand_shape, transpose, extract_slice,
/// etc.) that consume the loaded tensor into a single
/// `iree_linalg_ext.map_gather` op.
///
/// Note: tensor.pad is NOT supported for map_gather. Unlike map_scatter which
/// can write padding values directly to the output buffer (obtained from
/// store_to_buffer), map_gather reads from a source buffer and produces a
/// tensor result. There is no output memref available to write padding values
/// to at this stage.
LogicalResult combineLayoutTransformationForMapGather(
    MLIRContext *ctx, FunctionOpInterface funcOp, bool doReshapeByExpansion,
    CombineRelayoutOpsForGatherControlFnRef controlFn = nullptr);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATIONFORMAPGATHER_H_
