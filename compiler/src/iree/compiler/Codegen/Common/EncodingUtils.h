// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

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

/// Returns the encoding attribute from the type if there is an encoding.
/// Otherwise, returns null.
IREE::LinalgExt::EncodingAttr getEncodingAttr(RankedTensorType type);

/// Returns the original type that carried by encoding.
RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type);

/// Returns the RankedTensorType without encodings.
RankedTensorType dropEncoding(RankedTensorType type);

/// Returns the integer contained in an IntegerAttr, or zero if it has none.
int64_t getIntOrZero(IntegerAttr a);

/// Returns true if encoding user is one of matmul encodings.
bool isMatmulEncodingUser(IREE::LinalgExt::EncodingUser user);

/// Returns true if encoding user is one of batch matmul encodings.
bool isBatchMatmulEncodingUser(IREE::LinalgExt::EncodingUser user);

/// Returns true if the encoding is a vecmat.
bool isVecmatEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a matvec.
bool isMatvecEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a batch_vecmat.
bool isBatchVecmatEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a batch_matvec.
bool isBatchMatvecEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoded type is a vector.
bool isVectorEncoding(int64_t rank, IREE::LinalgExt::EncodingUser user);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo
getEncodingInfoForMatmul(IREE::LinalgExt::EncodingAttr encoding, int64_t rank,
                         TileMxNxK tileMxNxK);

MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr);

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

void populateMaterializeUpperBoundTileSizePatterns(
    RewritePatternSet &patterns, MaterializeEncodingFn materializeEncodingFn);

} // namespace iree_compiler
} // namespace mlir
#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
