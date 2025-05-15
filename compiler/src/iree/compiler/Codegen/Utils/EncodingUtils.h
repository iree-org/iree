// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler {

/// Extracts encoding info for materialization from the layout attribute of a
/// tensor type.
std::optional<IREE::Codegen::MaterializeEncodingInfo>
getEncodingInfoFromLayouts(RankedTensorType type);

/// Returns the inner tile sizes to be used for the given tensor type.
FailureOr<SmallVector<OpFoldResult>> getInnerTileSizesOfrImpl(
    OpBuilder &rewriter, Location loc, RankedTensorType tensorType,
    IREE::Encoding::LayoutAttrInterface layoutAttr,
    const IREE::Codegen::MaterializeEncodingInfo &materializeEncodingInfo);

/// Returns the materialized packed and swizzled shape for a
/// `dispatchTensorType` that binds a `RankedTensorType` with encoding. The
/// dynamic dimension sizes of the `dispatchTensorType` are provided in
/// `dynamicDims`.
FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensorImpl(
    OpBuilder &builder, Location loc,
    IREE::TensorExt::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims, IREE::Encoding::LayoutAttrInterface layoutAttr,
    IREE::Codegen::MaterializeEncodingInfo encodingInfo);

/// Applies an returns a tile-swizzling permutation to a packed shape.
SmallVector<OpFoldResult>
getSwizzledShape(ArrayRef<OpFoldResult> packedShape,
                 IREE::Codegen::MaterializeEncodingInfo encodingInfo);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_
