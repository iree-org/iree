// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

using MaterializeEncodingFn =
    std::function<FailureOr<IREE::Codegen::MaterializeEncodingInfo>(
        RankedTensorType, IREE::HAL::ExecutableTargetAttr targetAttr)>;

struct MaterializeEncodingValueInfo {
  SmallVector<Value> innerTileSizes;
};

using MaterializeEncodingValueFn =
    std::function<FailureOr<MaterializeEncodingValueInfo>(
        RankedTensorType, OpBuilder &, Location)>;

//===---------------------------------------------------------------------===//
// TypeConverter
//===---------------------------------------------------------------------===//

/// TypeConverter to use for materializing the encoding.
class MaterializeEncodingTypeConverter : public TypeConverter {
public:
  MaterializeEncodingTypeConverter(
      IREE::Codegen::LayoutAttrInterface layoutAttr,
      MaterializeEncodingValueFn materializeEncodingValueFn);

  const IREE::Codegen::LayoutAttrInterface &getLayoutAttr() const {
    return layoutAttr;
  }

  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(RankedTensorType type) const;

  FailureOr<SmallVector<OpFoldResult>> getInnerTileSizesOfr(
      OpBuilder &rewriter, Location loc, RankedTensorType tensorType,
      const IREE::Codegen::MaterializeEncodingInfo &materializeEncodingInfo)
      const;

private:
  const IREE::Codegen::LayoutAttrInterface layoutAttr;
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

/// Conversion target to use for for materializing the encoding.
struct MaterializeEncodingConversionTarget : public ConversionTarget {
  MaterializeEncodingConversionTarget(MLIRContext &context);
};

//===---------------------------------------------------------------------===//
// Utility methods about Encoding.
//===---------------------------------------------------------------------===//

/// Returns the deserialized MaterializeEncodingInfo if the `layouts` field is
/// present in encodings and it only has a single layout. Otherwise, returns
/// std::nullopt.
std::optional<IREE::Codegen::MaterializeEncodingInfo>
getEncodingInfoFromLayouts(RankedTensorType type);

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

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
