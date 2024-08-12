// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

/// Container of information needed to materialize the pack operation.
struct MaterializeEncodingInfo {
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;
  unsigned srcRank = 0;
};

using MaterializeEncodingFn =
    std::function<FailureOr<MaterializeEncodingInfo>(RankedTensorType)>;

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
struct MaterializeEncodingTypeConverter : public TypeConverter {
  MaterializeEncodingTypeConverter(MaterializeEncodingFn fn);
  const MaterializeEncodingFn &getMaterializeEncodingFn() const {
    return materializeEncodingFn;
  }

private:
  const MaterializeEncodingFn materializeEncodingFn;
};

/// Conversion target to use for for materializing the encoding.
struct MaterializeEncodingConversionTarget : public ConversionTarget {
  MaterializeEncodingConversionTarget(MLIRContext &context);
};

/// Base class for patterns that materialize encoding.
template <typename OpTy>
class OpMaterializeEncodingPattern : public OpConversionPattern<OpTy> {
public:
  OpMaterializeEncodingPattern(
      MLIRContext *context,
      const MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

//===---------------------------------------------------------------------===//
// Utility methods about Encoding.
//===---------------------------------------------------------------------===//

/// Returns the original type that carried by encoding.
RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type);

/// Returns the RankedTensorType without encodings.
RankedTensorType dropEncoding(RankedTensorType type);

/// Returns the integer contained in an IntegerAttr, or zero if it has none.
int64_t getIntOrZero(IntegerAttr a);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo
getEncodingInfoForMatmul(IREE::Encoding::EncodingAttr encoding, int64_t rank,
                         TileMxNxK tileMxNxK);

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// For now this takes a `paddingValue` as input. The source is also taken
/// as input so that these could be used with `OpConversionPatterns`.
FailureOr<tensor::PackOp> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, IREE::Encoding::SetEncodingOp encodingOp,
    Value source, MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Utility method to convert from `unset_encoding` op to `unpack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
FailureOr<tensor::UnPackOp> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Pouplates the set of patterns that lowers set_encoding, unset_encoding, and
/// upstream dialect ops with encoding types to pack/unpack ops.
void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Pouplates the set of patterns that lowers IREE dialect (e.g., Flow, Hal,
/// etc) ops with encoding types to pack/unpack ops.
void populateIREEMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

// Returns true if `encoding` represents a narrow-N matmul RESULT, e.g. the
// result of a matvec.
bool isNarrowNResult(IREE::Encoding::EncodingAttr encoding);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
