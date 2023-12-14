// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir {
namespace iree_compiler {

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
class GPULowerToUKernelsPass
    : public GPULowerToUKernelsBase<GPULowerToUKernelsPass> {
public:
  GPULowerToUKernelsPass(bool skipIntermediateRoundings)
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
getFnNameAndDefAttrs(const char *ukernelName, std::string &typeSuffixID,
                     RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isROCMBackend(targetAttr)) {
    result.name =
        std::string("__iree_uk_rocm_") + ukernelName + "_" + typeSuffixID;
    // TODO(#12327): Based on description in the issue, add an attribute
    // `vm.import.module` and set it to `vmvx`. This only works on `vmvx`
    // backend (obviously), but is enough to unblock while the proper fix
    // lands. For now there are a bunch of attributes set on the function, but
    // this should be made more controllable based on the backend.
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("rocm"));
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

static LogicalResult isArgmaxOp(linalg::GenericOp genericOp) {
  // Check for 2 results(value, index), and 1 input
  if (genericOp.getNumDpsInits() != 2) {
    return failure();
  }
  if (genericOp.getNumDpsInputs() != 1) {
    return failure();
  }

  // If max value is being used, it is not a pure argmax.
  if (!genericOp.getResults()[0].use_empty()) {
    return failure();
  }

  // Check that the rank is at least 3 and all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  // Currently only support 2D argmax to simplify problem.
  // Tiling pipeline is also set to tile all parallel dims to 1, and
  // reduction dim to be size of whole reduction problem. Which allow
  // this constraint to be true for a lot of argmax variances.
  if (numLoops > 2) {
    return failure();
  }
  // Argmax will require 1D reduction.
  if (numParallelLoops != (numLoops - 1)) {
    return failure();
  }
  // TODO: Add better affine map checks.
  auto indexing_maps = genericOp.getIndexingMapsArray();
  if (!indexing_maps[0].isIdentity())
    return failure();

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.select,
  // preceded by an arith.cmpf, arith.maximumf, and arith.extui
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield 1st arg is arith.maximumf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::MaximumFOp>())) {
      return failure();
    }
  }

  // Producer of linalg.yield op 2nd arg is arith.select
  // TODO: Add check that select is selecting between linalg.index and index of
  // current max.
  {
    producerOutput = yieldOp->getOperand(1);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::SelectOp>())) {
      return failure();
    }
  }

  // Producer of arith.select op is arith.cmpf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    auto producerCmpFOp = dyn_cast<arith::CmpFOp>(producer);
    if (!producerCmpFOp) {
      return failure();
    }
    if (producerCmpFOp.getPredicate() != arith::CmpFPredicate::OGT) {
      return failure();
    }

    // Check that in and out of cmpf are loop variables.
    // Currently first operand is disabled because it may be mixed type
    // which would lead it to be extf(%arg0).
    // TODO: Add better mixed type support check.
    if (producer->getOperand(1) != genericOp.getBody()->getArgument(1)) {
      return failure();
    }
  }

  return success();
}

/// Matches an (linalg.fill -> )? linalg.mmt4d operation sequence and converts
/// it into a iree_codegen.ukernel.mmt4d operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchArgmaxDAGForUKernel(RewriterBase &rewriter, linalg::GenericOp op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "argmax";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  // Get value/input type.
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = llvm::cast<ShapedType>(input.getType());
  Type inputElemType = inputType.getElementType();
  // Only support f16 and f32 values.
  if (!inputElemType.isF16() && !inputElemType.isF32()) {
    return failure();
  }

  // Get index type.
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = llvm::cast<ShapedType>(index.getType());
  Type indexElemType = indexType.getElementType();
  // Only support i32 and i64 index.
  if (!indexElemType.isInteger(32) && !indexElemType.isInteger(64)) {
    return failure();
  }

  std::string typeSuffixID = "";
  if (inputElemType.isF16() && indexElemType.isInteger(32)) {
    typeSuffixID = "F16I32";
  } else if (inputElemType.isF16() && indexElemType.isInteger(64)) {
    typeSuffixID = "F16I64";
  } else if (inputElemType.isF32() && indexElemType.isInteger(32)) {
    typeSuffixID = "F32I32";
  } else if (inputElemType.isF32() && indexElemType.isInteger(64)) {
    typeSuffixID = "F32I64";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  // Check if the accumulator is zero-filled.
  Value out = op.getDpsInitOperand(0)->get();
  if (!isInitializedToZero(out) && !isInitializedToZero(index)) {
    return rewriter.notifyMatchFailure(op, "Do not support accumulate");
  }

  Location loc = op.getLoc();
  // Currently only support 1D reduction, where reduc is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = op.getNumLoops() - 1;
  Value reductionDimSize =
      rewriter.create<tensor::DimOp>(loc, input, kReductionDim);
  auto fn =
      getFnNameAndDefAttrs(ukernelName, typeSuffixID, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, indexType, fn.name, ValueRange{input}, index,
      ValueRange{reductionDimSize},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;

struct LowerArgmaxToUKernelPattern : OpRewritePattern<linalg::GenericOp> {
  LowerArgmaxToUKernelPattern(MLIRContext *context,
                              TargetPredicate targetPredicate,
                              bool skipIntermediateRoundings = false)
      : OpRewritePattern<linalg::GenericOp>(context),
        targetPredicate(targetPredicate),
        skipIntermediateRoundings(skipIntermediateRoundings) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }
    if (failed(isArgmaxOp(op))) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchArgmaxDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    // llvm::outs()<<"SUCCESS, from:" <<op<<"\n";
    // llvm::outs()<<"SUCCESS, using:" <<ukernelOp<<"\n";
    rewriter.replaceAllUsesWith(op.getResults()[1],
                                ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  bool skipIntermediateRoundings;
};

} // namespace

void GPULowerToUKernelsPass::runOnOperation() {
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
  patterns.insert<LowerArgmaxToUKernelPattern>(context, isROCMBackend);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>>
createGPULowerToUKernelsPass(bool skipIntermediateRoundings) {
  return std::make_unique<GPULowerToUKernelsPass>(skipIntermediateRoundings);
}

} // namespace iree_compiler
} // namespace mlir
