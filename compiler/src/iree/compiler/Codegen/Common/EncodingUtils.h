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
      MaterializeEncodingFn fn, IREE::HAL::ExecutableTargetAttr targetAttr,
      bool transposeNarrowN, IREE::Codegen::LayoutAttrInterface layoutAttr);

  const IREE::Codegen::LayoutAttrInterface &getLayoutAttr() const {
    return layoutAttr;
  }

  const MaterializeEncodingFn &getMaterializeEncodingFn() const {
    return materializeEncodingFn;
  }

  IREE::HAL::ExecutableTargetAttr getTargetAttr() const { return targetAttr; }

  FailureOr<IREE::Codegen::MaterializeEncodingInfo>
  getEncodingInfo(RankedTensorType type) const {
    if (layoutAttr) {
      return layoutAttr.getEncodingInfo(type);
    }
    return materializeEncodingFn(type, targetAttr);
  }

  bool getTransposeNarrowN() const { return transposeNarrowN; }

private:
  const MaterializeEncodingFn materializeEncodingFn;
  const IREE::HAL::ExecutableTargetAttr targetAttr;
  bool transposeNarrowN = false;
  // The `layoutAttr` implements the logics of encoding materialization. It has
  // a higher priority when it presents.
  // TODO(hanchung): Move the logics that take `targetAttr` and
  // `transposeNarrowN` into accounts to their own attribute implementation. It
  // is in a transition state, so we have two path atm. We're incrementally
  // moving the logics to attributes.
  const IREE::Codegen::LayoutAttrInterface layoutAttr;
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

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

IREE::Codegen::MaterializeEncodingInfo
getEncodingInfoForMatmul(IREE::Encoding::EncodingAttr encoding, int64_t rank,
                         TileMxNxK tileMxNxK);

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// Note that `source` could be returned when the pack op is a nop. Because
/// creating a new operation is not cheap.
/// TODO(hanchung): Move the utility to
/// `Dialect/Codegen/Utils/LayoutUtils.[h|cpp]`. It is not moved because it
/// needs some cleanup for the c++ file. E.g., the method will no longer take
/// the type converter into account. Ideally we should move CPU specific
/// patterns (e.g., lowerContractionOpWithEncoding, etc) to their LayoutAttr
/// implementation; move general patterns to the utilitiy files, and retire this
/// file.
FailureOr<Value> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, IREE::Encoding::SetEncodingOp encodingOp,
    Value source, const MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Utility method to convert from `unset_encoding` op to `unpack` operation.
/// Note that `packedValue` could be returned when the unpack op is a nop.
/// Because creating a new operation is not cheap.
/// TODO(hanchung): Move the implementation to
/// `Dialect/Codegen/Utils/LayoutUtils.[h|cpp]`. See the reason in the function
/// comment of lowerSetEncodingToPackOp method.
FailureOr<Value> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, const MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Pouplates the set of patterns that lowers set_encoding, unset_encoding, and
/// upstream dialect ops with encoding types to pack/unpack ops.
void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

/// Pouplates the set of patterns that lowers shape-like operations (e.g., Flow
/// ops, Hal ops, tensor.empty, linalg.fill, etc) with encoding types to the
/// same op with materialized shapes.
void populateShapeIndependentMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn);

// Returns true if `encoding` represents a narrow-N matmul RESULT, e.g. the
// result of a matvec.
bool isNarrowNResult(IREE::Encoding::EncodingAttr encoding);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGUTILS_H_
