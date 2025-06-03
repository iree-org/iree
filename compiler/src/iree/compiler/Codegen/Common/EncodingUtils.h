// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// TypeConverter
//===---------------------------------------------------------------------===//

/// TypeConverter to use for materializing the encoding.
class MaterializeEncodingTypeConverter : public TypeConverter {
public:
  MaterializeEncodingTypeConverter(
      IREE::Encoding::LayoutMaterializerAttr layoutAttr);

  const IREE::Encoding::LayoutMaterializerAttr &getLayoutAttr() const {
    return layoutAttr;
  }

  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(RankedTensorType type) const;

  /// Returns the inner tile sizes to be used for the given tensor type.
  FailureOr<SmallVector<OpFoldResult>> getInnerTileSizesOfr(
      OpBuilder &rewriter, Location loc, RankedTensorType tensorType,
      const IREE::Codegen::MaterializeEncodingInfo &materializeEncodingInfo)
      const;

  /// Returns the materialized packed and swizzled shape for a
  /// `dispatchTensorType` that binds a `RankedTensorType` with encoding. The
  /// dynamic dimension sizes of the `dispatchTensorType` are provided in
  /// `dynamicDims`.
  FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensor(
      OpBuilder &builder, Location loc,
      IREE::TensorExt::DispatchTensorType dispatchTensorType,
      ValueRange dynamicDims) const;

  /// Returns success if materialized `newOffsets`, `newSizes` and `newStrides`
  /// can be calculated and set for the slice specified by `offsets`, `sizes`
  /// and `strides` on the dispatch tensor `type` with potential `dynamicDims`
  /// sizes.
  LogicalResult getOffsetsSizesStrides(
      OpBuilder &builder, Location loc,
      IREE::TensorExt::DispatchTensorType type, ValueRange dynamicDims,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      ArrayRef<OpFoldResult> strides, SmallVectorImpl<OpFoldResult> &newOffsets,
      SmallVectorImpl<OpFoldResult> &newSizes,
      SmallVectorImpl<OpFoldResult> &newStrides) const;

private:
  const IREE::Encoding::LayoutMaterializerAttr layoutAttr;
};

/// Conversion target to use for for materializing the encoding.
struct MaterializeEncodingConversionTarget : public ConversionTarget {
  MaterializeEncodingConversionTarget(MLIRContext &context);
};

//===---------------------------------------------------------------------===//
// Utility methods about Encoding.
//===---------------------------------------------------------------------===//

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// NOTE: `source` could be returned when packing is not needed.
FailureOr<Value> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, IREE::Encoding::SetEncodingOp encodingOp,
    Value source, const MaterializeEncodingTypeConverter &typeConverter);

/// Utility method to convert from `unset_encoding` op to `unpack` operation.
/// NOTE: `packedValue` could be returned when unpacking is not needed.
FailureOr<Value> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, const MaterializeEncodingTypeConverter &typeConverter);

/// Populates the set of patterns that lowers operations with encoding types to
/// operations without encodings.
void populateMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter);

void populateLoadStoreMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
