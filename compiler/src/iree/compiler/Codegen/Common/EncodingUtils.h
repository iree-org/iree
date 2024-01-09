// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

/// Returns the encoding attribute from the type if there is an encoding.
/// Otherwise, returns null.
IREE::LinalgExt::EncodingAttr getEncodingAttr(RankedTensorType type);

/// Get the permutation that permutes the input shape to the canonical
/// matmul input shape based on the IndexingMaps encoding attribute.
std::optional<SmallVector<int64_t>>
getPermutationToCanonicalMatmulShape(IREE::LinalgExt::EncodingAttr encoding);

/// Returns a RankedTensorType that has been transposed into the canonical
/// form for an ordinary matmul/batch_matmul op.
RankedTensorType getCanonicalMatmulTypeWithEncoding(RankedTensorType type);

/// Returns the ContractionDimensions for the encoding user_indexing_maps
FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(IREE::LinalgExt::EncodingAttr encoding);

/// Returns the original type that carried by encoding.
RankedTensorType getOriginalTypeWithEncoding(RankedTensorType type);

/// Returns the RankedTensorType without encodings.
RankedTensorType dropEncoding(RankedTensorType type);

/// Returns the integer contained in an IntegerAttr, or zero if it has none.
int64_t getIntOrZero(IntegerAttr a);

/// Returns true if the encoding is a vecmat.
bool isVecmatEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a matvec.
bool isMatvecEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a batch_vecmat.
bool isBatchVecmatEncoding(IREE::LinalgExt::EncodingAttr encoding);

/// Returns true if the encoding is a batch_matvec.
bool isBatchMatvecEncoding(IREE::LinalgExt::EncodingAttr encoding);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo
getEncodingInfoForMatmul(IREE::LinalgExt::EncodingAttr encoding, int64_t rank,
                         TileMxNxK tileMxNxK);

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

void populateMaterializeUpperBoundTileSizePatterns(
    RewritePatternSet &patterns, MaterializeEncodingFn materializeEncodingFn);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
