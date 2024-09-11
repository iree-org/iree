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

/// Container of information needed to materialize the layout transformations.
///
/// On CPU, these layout transformations consist of a single `temsor.pack`
/// or `tensor.unpack` op, implementing a tiled layout where each tile is
/// row-major.
///
/// On GPU, there is an additional `swizzle`, which changes the layout inside
/// of the tile. See the comment on the nested Swizzle struct.
struct MaterializeEncodingInfo {
  // Metadata for a swizzle, that is, an (expand_shape -> transposition)
  // pair of ops performing a change of layout within the tiles. This is used
  // on GPU, where the tiles themselves can have an arbitrary layout.
  struct Swizzle {
    // This vector-of-vectors contains all the information needed to generate
    // a `tensor.expand_shape` creating additional internal dimensions into the
    // tile. For example, expandShape = [[16], [4, 2]] means that the original
    // tile shape [16, 8] gets expanded such that the first dimension 16 is left
    // unchanged, and the second dimension 8 gets split into two internal dims
    // of size 4 and 2.
    SmallVector<SmallVector<int64_t>> expandShape;
    // This permutation vector applies to the expanded dimensions and is used
    // to generate a `linalg.transpose` changing the layout of the tile. For
    // example, permutation[0] dictates which of the expanded dimensions becomes
    // the leading dimension of the layout.
    SmallVector<int64_t> permutation;
  };

  // The next 3 fields are used to create a `tensor.pack` or `tensor.unpack` op,
  // changing the overall layout between row-major and tiled (where each tile is
  // row-major).
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;

  // The optional swizzle, see the above comment on Swizzle. Only used on GPU.
  std::optional<Swizzle> swizzle;
};

using MaterializeEncodingFn = std::function<FailureOr<MaterializeEncodingInfo>(
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
  MaterializeEncodingTypeConverter(MaterializeEncodingFn fn,
                                   IREE::HAL::ExecutableTargetAttr targetAttr);

  const MaterializeEncodingFn &getMaterializeEncodingFn() const {
    return materializeEncodingFn;
  }

  IREE::HAL::ExecutableTargetAttr getTargetAttr() const { return targetAttr; }

  FailureOr<MaterializeEncodingInfo>
  getEncodingInfo(RankedTensorType type) const {
    return materializeEncodingFn(type, targetAttr);
  }

private:
  const MaterializeEncodingFn materializeEncodingFn;
  const IREE::HAL::ExecutableTargetAttr targetAttr;
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
    Value source, const MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Utility method to convert from `unset_encoding` op to `unpack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
FailureOr<tensor::UnPackOp> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, const MaterializeEncodingTypeConverter &typeConverter,
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

// Concatenates the vectors.
SmallVector<int64_t>
getExpandedTileShape(SmallVector<SmallVector<int64_t>> expandShape);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
