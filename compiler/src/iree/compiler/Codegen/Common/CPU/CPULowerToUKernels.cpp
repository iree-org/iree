// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Common/CPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

// Returns the CastOpInterface op of the body, if
//   - the `genericOp` is element-wise with identity maps, and
//   - it has only a  CastOpInterface op.
// Returns std::nullopt, otherwise.
static std::optional<CastOpInterface>
getCastOpOfElementWiseCast(linalg::GenericOp genericOp) {
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  auto castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return std::nullopt;
  }
  Value castIn = castOp->getOperand(0);
  if (castIn.isa<BlockArgument>() &&
      castIn.cast<BlockArgument>().getArgNumber() != 0) {
    return std::nullopt;
  }
  return castOp;
}

namespace {
class CPULowerToUKernelsPass
    : public CPULowerToUKernelsBase<CPULowerToUKernelsPass> {
public:
  CPULowerToUKernelsPass(bool skipIntermediateRoundings)
      : skipIntermediateRoundings(skipIntermediateRoundings) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // This option defaults to `true` both in Passes.td and in C++ code.
    // If either side has `false`, that's a non-default choice, so we let that
    // override a `true` on the other side.
    skipIntermediateRoundings &= optionSkipIntermediateRoundings;
    return success();
  }

private:
  bool skipIntermediateRoundings;
};
} // namespace

/// Returns `true` if an `outsOperand` value is initialized to zero.
static bool isInitializedToZero(Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return false;
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, m_Zero()) ||
         matchPattern(fillVal, m_AnyZeroFloat());
}

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isVMVXBackend(targetAttr)) {
    result.name = std::string("vmvx.") + ukernelName;
    // TODO(#12327): Based on description in the issue, add an attribute
    // `vm.import.module` and set it to `vmvx`. This only works on `vmvx`
    // backend (obviously), but is enough to unblock while the proper fix
    // lands. For now there are a bunch of attributes set on the function, but
    // this should be made more controllable based on the backend.
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("vmvx"));
  } else {
    result.name = std::string("iree_uk_") + ukernelName;
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.fields"),
        rewriter.getArrayAttr({rewriter.getStringAttr("processor_data")}));
    result.defAttrs.emplace_back(rewriter.getStringAttr("hal.import.bitcode"),
                                 rewriter.getBoolAttr(true));
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.cconv"),
        IREE::HAL::CallingConventionAttr::get(
            rewriter.getContext(),
            IREE::HAL::CallingConvention::ParameterStruct));
  }
  return result;
}

// If the defining op of `input` is an element-wise cast, return the input to
// the casting `linalg.generic` op. Otherwise, return `input`.
static Value getInputForUKernel(Value input) {
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  std::optional<CastOpInterface> castOp = getCastOpOfElementWiseCast(genericOp);
  if (!castOp) {
    return input;
  }
  return genericOp->getOperand(0);
}

// If the defining op of `input` is an element-wise cast, return the element
// type of the cast source with explicit signedness. Otherwise, return the
// element type of `input`.
static Type getElementTypeForUKernel(Value input) {
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  std::optional<CastOpInterface> castOp = getCastOpOfElementWiseCast(genericOp);
  if (!castOp) {
    return llvm::cast<ShapedType>(input.getType()).getElementType();
  }
  Type castOpSrcType = castOp.value()->getOperand(0).getType();
  if (isa<arith::ExtUIOp>(*castOp)) {
    return IntegerType::get(castOp->getContext(),
                            castOpSrcType.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return castOpSrcType;
}

/// Matches an (linalg.fill -> )? linalg.mmt4d operation sequence and converts
/// it into a iree_codegen.ukernel.mmt4d operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::Mmt4DOp op,
                   bool skipIntermediateRoundings) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "mmt4d";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  Value lhs = getInputForUKernel(op.getDpsInputOperand(0)->get());
  Value rhs = getInputForUKernel(op.getDpsInputOperand(1)->get());
  Value out = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = getElementTypeForUKernel(op.getDpsInputOperand(0)->get());
  Type rhsElemType = getElementTypeForUKernel(op.getDpsInputOperand(1)->get());
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_S8S8S32;
  } else if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(4) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_S8S4S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isSignlessInteger(16) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_S16S16S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isUnsignedInteger(4) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_S16U4S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isSignlessInteger(8) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_S16S8S32;
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F32F32F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F16F16F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF16()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F16F16F16;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isBF16()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  // Check if the accumulator is zero-filled.
  if (isInitializedToZero(out)) {
    // Not setting flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE, so the mmt4d op won't
    // read the existing accumulator, so its defining op can be discarded.
    if (auto fillOp = out.getDefiningOp<linalg::FillOp>()) {
      out = fillOp.getDpsInitOperand(0)->get();
    }
  } else {
    // Tell the mmt4d op to read the existing accumulator.
    flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
  }

  if (skipIntermediateRoundings) {
    flags |= IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS;
  }

  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 0);
  Value k = rewriter.create<tensor::DimOp>(loc, rhs, 1);

  auto getDimAsI32 = [](RewriterBase &rewriter, Location loc, Value value,
                        int dim) -> Value {
    return rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, dim));
  };
  Value m0 = getDimAsI32(rewriter, loc, lhs, 2);
  Value n0 = getDimAsI32(rewriter, loc, rhs, 2);
  Value k0 = getDimAsI32(rewriter, loc, rhs, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto fn = getFnNameAndDefAttrs(ukernelName, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, m0, n0, k0, flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, tensor::PackOp op,
                   bool /*skipIntermediateRoundings*/) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "pack";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = llvm::cast<ShapedType>(in.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(8)) {
    flags = IREE_UK_FLAG_PACK_TYPE_I8I8;
  } else if (inElemType.isSignlessInteger(32) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_PACK_TYPE_I32I32;
  } else if (inElemType.isF32() && outElemType.isF32()) {
    flags = IREE_UK_FLAG_PACK_TYPE_F32F32;
  } else if (inElemType.isF16() && outElemType.isF16()) {
    flags = IREE_UK_FLAG_PACK_TYPE_F16F16;
  } else if (inElemType.isBF16() && outElemType.isBF16()) {
    flags = IREE_UK_FLAG_PACK_TYPE_BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected input to be 2D");
  }

  if (outType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected output to be 4D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Type i64 = rewriter.getI64Type();

  // The ukernel requires a padding value of type i64. When the element type is
  // a narrower N-bit type, only the least significant N bits of the i64 padding
  // value are used.
  Value paddingVal = op.getPaddingValue();
  // If the pack op didn't have a padding_value attribute, default to 0.
  if (!paddingVal) {
    paddingVal =
        rewriter.create<arith::ConstantOp>(loc, i64, rewriter.getZeroAttr(i64));
  }
  int paddingValBitWidth = paddingVal.getType().getIntOrFloatBitWidth();
  // Non-integer element types get bitcast to integer of same bit width.
  if (!paddingVal.getType().isSignlessInteger()) {
    Type sameWidthIntType = rewriter.getIntegerType(paddingValBitWidth);
    if (!sameWidthIntType) {
      return rewriter.notifyMatchFailure(op, "no integer type with this width");
    }
    paddingVal =
        rewriter.create<arith::BitcastOp>(loc, sameWidthIntType, paddingVal);
  }
  // Element types > 64bits could be supported, when the padding value is a
  // repeating 64-bit pattern. For now, we leave this as not-yet-implemented.
  if (paddingValBitWidth > 64) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported padding_value bit width");
  }
  // Integers narrower than 64 bit get extended to 64 bits, it doesn't matter
  // how, as the high bits are unused.
  if (paddingValBitWidth < 64) {
    paddingVal = rewriter.create<arith::ExtUIOp>(loc, i64, paddingVal);
  }
  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value out_size2 = rewriter.create<tensor::DimOp>(loc, out, 2);
  Value out_size3 = rewriter.create<tensor::DimOp>(loc, out, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto fn = getFnNameAndDefAttrs(ukernelName, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, in, out,
      ValueRange{in_size0, in_size1, out_size0, out_size1, out_size2, out_size3,
                 paddingVal, flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, tensor::UnPackOp op,
                   bool /*skipIntermediateRoundings*/) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "unpack";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = llvm::cast<ShapedType>(in.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (inElemType.isSignlessInteger(32) && outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_I32I32;
  } else if (inElemType.isF32() && outElemType.isF32()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_F32F32;
  } else if (inElemType.isF16() && outElemType.isF16()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_F16F16;
  } else if (inElemType.isBF16() && outElemType.isBF16()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected input to be 4D");
  }

  if (outType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected output to be 2D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value in_size2 = rewriter.create<tensor::DimOp>(loc, in, 2);
  Value in_size3 = rewriter.create<tensor::DimOp>(loc, in, 3);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto fn = getFnNameAndDefAttrs(ukernelName, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, in, out,
      ValueRange{in_size0, in_size1, in_size2, in_size3, out_size0, out_size1,
                 flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static uint32_t
getFlagForUserAndOperandTypes(IREE::LinalgExt::EncodingAttr encoding,
                              ArrayRef<Attribute> operandTypes) {
  // There are currently no batch_mmt4d ukernels, so check for no batch
  // dimension.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || !cDims->batch.empty() || operandTypes.size() != 3) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_NONE;
  }

  Type lhs = operandTypes[0].cast<TypeAttr>().getValue();
  Type rhs = operandTypes[1].cast<TypeAttr>().getValue();
  Type out = operandTypes[2].cast<TypeAttr>().getValue();

  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32;
  } else if (lhs.isF16() && rhs.isF16() && out.isF32()) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F32;
  } else if (lhs.isF16() && rhs.isF16() && out.isF16()) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F16;
  } else if (lhs.isBF16() && rhs.isBF16() && out.isF32()) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16F32;
  } else if (lhs.isBF16() && rhs.isBF16() && out.isBF16()) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16BF16;
  } else if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
             out.isSignlessInteger(32)) {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
  } else {
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_NONE;
  }
}

static uint32_t getFlagForRole(IREE::LinalgExt::EncodingRole role) {
  switch (role) {
  case IREE::LinalgExt::EncodingRole::LHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS;
  case IREE::LinalgExt::EncodingRole::RHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS;
  case IREE::LinalgExt::EncodingRole::RESULT:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT;
  default:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_NONE;
  }
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, IREE::Codegen::QueryTileSizesOp op,
                   bool /*skipIntermediateRoundings*/) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "query_tile_sizes.2d";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  auto tensorType = op.getTensorType().dyn_cast<RankedTensorType>();
  if (!tensorType) {
    return rewriter.notifyMatchFailure(op,
                                       "need a ranked tensor type attribute");
  }
  if (tensorType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "only the 2D case is implemented");
  }
  auto encoding = tensorType.getEncoding()
                      .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (!encoding) {
    return rewriter.notifyMatchFailure(op, "no encoding attribute");
  }
  SmallVector<Type> resultTypes(tensorType.getRank(), rewriter.getIndexType());
  SmallVector<Value> inputValues;
  Location loc = op.getLoc();
  for (int64_t i : tensorType.getShape()) {
    inputValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
  }
  uint32_t flagForUserAndOperandTypes = getFlagForUserAndOperandTypes(
      encoding, encoding.getElementTypes().getValue());
  uint32_t flagForRole = getFlagForRole(encoding.getRole().getValue());
  if (!flagForUserAndOperandTypes || !flagForRole) {
    return rewriter.notifyMatchFailure(op, "unhandled encoding");
  }
  inputValues.push_back(rewriter.create<arith::ConstantIntOp>(
      loc, flagForUserAndOperandTypes | flagForRole, 32));
  auto fn = getFnNameAndDefAttrs(ukernelName, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, resultTypes, fn.name, inputValues, /*outs=*/ValueRange{},
      /*other_operands=*/ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/IntegerAttr{});
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  LowerToUKernelPattern(MLIRContext *context, TargetPredicate targetPredicate,
                        bool skipIntermediateRoundings = false)
      : OpRewritePattern<OpType>(context), targetPredicate(targetPredicate),
        skipIntermediateRoundings(skipIntermediateRoundings) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchDAGForUKernel(rewriter, op, skipIntermediateRoundings);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  bool skipIntermediateRoundings;
};

} // namespace

void CPULowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.

  // These patterns are unconditionally enabled, because we have strong evidence
  // that it is difficult for codegen to consistently approach microkernels
  // performance, and that consideration overrides the benefit of fusions for
  // these ops.
  auto allTargets = [](auto target) { return true; };
  patterns.insert<LowerToUKernelPattern<linalg::Mmt4DOp>>(
      context, allTargets, skipIntermediateRoundings);
  // These patterns could in principle be used on LLVMCPU, not just VMVX, but
  // we choose not to, for two reasons:
  // 1. Codegen for these ops is thought to be good enough, that we do not
  //    really need microkernels.
  // 2. Fusions matter particularly for pack/unpack ops due to the memcpy-like
  //    nature of these ops and the typical patterns ML workload offering
  //    fusions opportunities there. The combinatorics of what could get fused
  //    there seem unbounded, so any attempt to add microkernels for fusions
  //    would have difficulty scaling.
  patterns.insert<LowerToUKernelPattern<tensor::PackOp>,
                  LowerToUKernelPattern<tensor::UnPackOp>>(context,
                                                           isVMVXBackend);
  // These patterns are inherently specific to the VMVX backend.
  patterns.insert<LowerToUKernelPattern<IREE::Codegen::QueryTileSizesOp>>(
      context, isVMVXBackend);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>>
createCPULowerToUKernelsPass(bool skipIntermediateRoundings) {
  return std::make_unique<CPULowerToUKernelsPass>(skipIntermediateRoundings);
}

} // namespace mlir::iree_compiler
