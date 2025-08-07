// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

/// Fill in the operands that are nullopt, such that `op` applied to `operands`
/// results in `output`. This is a sort of reverse constant folding. It is used
/// to propagate the required padding values from a linalg.generic op's
/// terminator to it's operands.
static bool completeOperandValues(Operation *op,
                                  SmallVector<std::optional<float>> &operands,
                                  float output) {

  if (isa<arith::MulFOp, arith::MulIOp>(op)) {

    // If output is zero, set all unset operands to zero.
    if (output == 0.0f) {
      for (auto &operand : operands) {
        if (!operand.has_value())
          operand = 0.0f;
      }
    }

    else {
      float productOfSetValues = 1.0f;
      for (auto &operand : operands) {
        if (operand.has_value())
          productOfSetValues *= operand.value();
      }

      // We are in the case where output is not zero. If there is a set value of
      // zero, it is impossible to complete the operands to give a product that
      // is non-zero.
      if (productOfSetValues == 0) {
        op->emitRemark()
            << "zero operand and non-zero result: impossible product";
        return false;
      }

      // Set all unset values. Set the first unset value such that the product
      // is as requred, and all subsequent unset values to 1.0f.
      bool complete = false;
      for (auto &operand : operands) {
        if (!operand.has_value()) {
          operand = complete ? output / productOfSetValues : 1.0f;
          complete = true;
        }
      }
    }

    // Check that the product of all operands is equal to the value.
    if (output != std::accumulate(operands.begin(), operands.end(), 1.0f,
                                  [](float a, std::optional<float> b) {
                                    return a * b.value();
                                  })) {
      op->emitRemark() << "failed to complete the operands to obtain product "
                       << output << ".";
      return false;
    }

    return true;
  }

  if (isa<arith::AddFOp, arith::AddIOp>(op)) {

    float sumOfSetValues = 0.0f;
    for (auto &operand : operands) {
      if (operand.has_value())
        sumOfSetValues += operand.value();
    }

    // Set the first unset value so as to obtain the required sum, and all
    // subsequent unset values to zero.
    bool complete = false;
    for (auto &operand : operands) {
      if (!operand.has_value()) {
        operand = complete ? output - sumOfSetValues : 0.0f;
        complete = true;
      }
    }

    // Check that the sum of all operands is equal to the value.
    if (output != std::accumulate(operands.begin(), operands.end(), 0.0f,
                                  [](float a, std::optional<float> b) {
                                    return a + b.value();
                                  })) {
      op->emitRemark() << "failed to complete the operands to obtain sum "
                       << output << ".";
      return false;
    }
    return true;
  }

  if (isa<arith::SubFOp, arith::SubIOp>(op)) {
    assert(operands.size() == 2 &&
           "subtraction should have exactly two operands");

    // a - b = output ==> b = a - output
    if (operands[0].has_value() && !operands[1].has_value()) {
      operands[1] = operands[0].value() - output;
    }

    // a - b = output ==> a = output + b
    else if (operands[1].has_value() && !operands[0].has_value()) {
      operands[0] = output + operands[1].value();
    }

    // a - b = output ==> a = output, b = 0 (one possible solution).
    else if (!operands[0].has_value() && !operands[1].has_value()) {
      operands[0] = output;
      operands[1] = 0.0f;
    }

    if (operands[0].value() - operands[1].value() != output) {
      op->emitRemark() << "failed to complete the operands to obtain " << output
                       << " from subtraction.";
      return false;
    }
    return true;
  }

  if (isa<math::ExpOp>(op)) {
    // log base e of the output
    auto lnOutput = std::log(output);
    assert(operands.size() == 1);
    if (operands[0].has_value() && operands[0].value() != lnOutput) {
      op->emitRemark() << "e^(" << operands[0].value() << ") != " << output
                       << ".";
      return false;
    }
    operands[0] = lnOutput;
    return true;
  }

  if (isa<arith::TruncFOp>(op)) {
    assert(operands.size() == 1);
    if (operands[0].has_value() && operands[0].value() != output) {
      op->emitRemark() << "trunc(" << operands[0].value() << ") != " << output
                       << ".";
      return false;
    }
    operands[0] = output;
    return true;
  }

  op->emitRemark() << "unhandled operation type.";
  return false;
};

static FailureOr<SmallVector<Attribute>>
getPaddingValues(TilingInterface tilingInterfaceOp, RewriterBase &rewriter) {

  Operation *op = tilingInterfaceOp.getOperation();

  // OnlineAttention mask padding needs to be mindful of softmax (pad with
  // -inf)
  // TODO: Extract this special case logic into an upstream
  // PaddingOpInterface.
  if (auto onlineAttentionOp =
          dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(op)) {

    SmallVector<Attribute> paddingValues;
    paddingValues.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      paddingValues.push_back(
          rewriter.getZeroAttr(getElementTypeOrSelf(operand.getType())));
    }

    TypedValue<ShapedType> mask = onlineAttentionOp.getMask();
    if (!mask) {
      tilingInterfaceOp.emitRemark(
          "failed to pad op: requires a mask operand to pad to the "
          "proper value. Consider materializing the mask operand "
          "explicitly.");
      return failure();
    }

    Type maskEltType = getElementTypeOrSelf(mask.getType());
    if (!llvm::isa<FloatType>(maskEltType)) {
      tilingInterfaceOp.emitRemark(
          "failed to pad op: -inf requires a float type");
      return failure();
    }
    int64_t idx = onlineAttentionOp.getMaskMutable()
                      .getAsOperandRange()
                      .getBeginOperandIndex();
    const auto &fltSemantics = cast<FloatType>(maskEltType).getFloatSemantics();
    paddingValues[idx] = rewriter.getFloatAttr(
        maskEltType, APFloat::getInf(fltSemantics, /*Negative=*/true));

    return paddingValues;
  }

  if (linalg::GenericOp genericOp = dyn_cast<mlir::linalg::GenericOp>(op)) {

    constexpr const float kInf = std::numeric_limits<float>::infinity();

    SmallVector<Attribute> paddingValues;
    Block *block = genericOp.getBlock();
    DenseMap<Value, float> paddingMap;

    Operation *yieldOp = block->getTerminator();
    if (yieldOp->getNumOperands() != 1) {
      tilingInterfaceOp.emitRemark("failed to handle yield 0 or 2+ operands");
      return failure();
    }

    Operation *reductionOp = yieldOp->getOperand(0).getDefiningOp();
    if (!reductionOp) {
      tilingInterfaceOp.emitRemark(
          "failed to handle yield without defining op");
      return failure();
    }

    auto reductionIdentityValue = [&]() -> FailureOr<float> {
      // x = x * 1.0
      if (isa<arith::MulFOp, arith::MulIOp>(reductionOp))
        return 1.0f;

      // x = x + 0.0
      if (isa<arith::AddFOp, arith::AddIOp>(reductionOp))
        return 0.0f;

      // x = max(x, -inf)
      if (isa<arith::MaxSIOp, arith::MaxUIOp, arith::MaxNumFOp>(reductionOp))
        return -kInf;

      // x = min(x, +inf)
      if (isa<arith::MinSIOp, arith::MinUIOp, arith::MinNumFOp>(reductionOp))
        return kInf;

      // Something else that needs to be handled
      reductionOp->emitRemark(
          "failed to determine reduction type from yield's defining op");
      return failure();
    }();
    if (failed(reductionIdentityValue)) {
      return failure();
    }

    // Perform a depth-first search, starting at the generic operation's
    // terminator and ending at its operands.
    SmallVector<Value> toProcess;
    for (Value operand : reductionOp->getOperands()) {
      paddingMap[operand] = reductionIdentityValue.value();
      toProcess.push_back(operand);
    }

    while (!toProcess.empty()) {
      Value valueToProcess = toProcess.back();
      toProcess.pop_back();
      auto opToProcess = valueToProcess.getDefiningOp();

      // Block argument:
      if (!opToProcess)
        continue;

      // Try and set the values of the operands of `opToProcess`. Some operands
      // may already be set, in which case the remaining operands must be
      // chosen to complete the equation.
      SmallVector<std::optional<float>> operandVals(
          opToProcess->getNumOperands(), std::nullopt);
      for (auto operand : llvm::enumerate(opToProcess->getOperands())) {
        Value value = operand.value();
        auto iter = paddingMap.find(value);
        if (iter != paddingMap.end()) {
          operandVals[operand.index()] = iter->second;
        }
      }
      bool complete = completeOperandValues(opToProcess, operandVals,
                                            paddingMap[valueToProcess]);
      if (!complete)
        return failure();

      for (auto operand : llvm::enumerate(opToProcess->getOperands())) {
        Value value = operand.value();
        if (paddingMap.find(value) == paddingMap.end()) {
          float newVal = operandVals[operand.index()].value();
          paddingMap[value] = newVal;
          toProcess.push_back(value);
        }
      }
    }

    // The depth-first search should now have propagated the values all the way
    // to the operands. Convert from float to the correct attribute type.
    for (auto operandIter :
         llvm::enumerate(tilingInterfaceOp.getOperation()->getOperands())) {
      auto index = operandIter.index();
      auto blockArgs = block->getArguments();
      assert(blockArgs.size() > index && "invalid index");
      BlockArgument blockArg = blockArgs[index];

      auto iter = paddingMap.find(blockArg);
      if (iter == paddingMap.end()) {
        tilingInterfaceOp.emitRemark("operand's padding value not inferred");
        return failure();
      }

      Type eltType = getElementTypeOrSelf(operandIter.value().getType());
      float padValue = iter->second;

      // Convert the float to the appropriate Attribute
      auto padAttr = [&]() -> Attribute {
        if (padValue == 0)
          return rewriter.getZeroAttr(eltType);
        if (padValue == 1)
          return rewriter.getOneAttr(eltType);

        bool isPosInf = (padValue == kInf);
        bool isNegInf = (padValue == -kInf);
        if ((isPosInf || isNegInf) && !llvm::isa<FloatType>(eltType)) {
          tilingInterfaceOp.emitRemark("-inf/+inf requires a float type");
          return {};
        }

        if (isPosInf || isNegInf) {
          const auto &fltSemantics =
              cast<FloatType>(eltType).getFloatSemantics();
          return rewriter.getFloatAttr(eltType,
                                       APFloat::getInf(fltSemantics,
                                                       /*Negative=*/isNegInf));
        }

        tilingInterfaceOp.emitRemark("failed to get padding attribute from ")
            << padValue << ".";
        return {};
      }();

      if (!padAttr)
        return failure();

      paddingValues.push_back(padAttr);
    }

    return paddingValues;
  }

  tilingInterfaceOp.emitRemark("Failed to determine padding values");
  return failure();
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {

  // 2. Get padding sizes from tileSizes.
  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);
  SmallVector<OpFoldResult> padSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  auto paddingValues = getPaddingValues(tilingInterfaceOp, rewriter);
  if (failed(paddingValues))
    return failure();

  // 3. Set options.
  auto options = linalg::PadTilingInterfaceOptions()
                     .setPaddingSizes(padSizes)
                     .setPaddingValues(paddingValues.value())
                     .setPadToMultipleOf(true);

  LLVM_DEBUG(DBGS() << "Start padding " << *tilingInterfaceOp << "\n";
             DBGS() << "--with tile sizes: "
                    << llvm::interleaved_array(options.paddingSizes) << "\n";
             DBGS() << "--with padding values: "
                    << llvm::interleaved_array(options.paddingValues) << "\n";
             DBGS() << "--with padToMultipleOf: " << options.padToMultipleOf
                    << "\n");

  // 4. Pad.
  SmallVector<tensor::PadOp> padOps;
  FailureOr<TilingInterface> maybePaddedOp =
      linalg::rewriteAsPaddedOp(rewriter, tilingInterfaceOp, options, padOps);
  if (failed(maybePaddedOp)) {
    tilingInterfaceOp.emitWarning("failed to pad op");
    return failure();
  }

  // 5. For each PadOp, create a linalg::CopyOp to allow dim propagations.
  TilingInterface paddedOp = *maybePaddedOp;
  for (auto padOp : padOps) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(padOp);

    // Record users for RAUW before creating new users.
    llvm::SmallDenseSet<Operation *> users(padOp.getResult().getUsers().begin(),
                                           padOp.getResult().getUsers().end());

    RankedTensorType tensorTy = padOp.getResultType();
    int64_t rank = tensorTy.getRank();
    SmallVector<OpFoldResult> sizes(rank, OpFoldResult());
    for (int64_t i = 0; i < rank; ++i) {
      sizes[i] = rewriter.createOrFold<tensor::DimOp>(paddedOp->getLoc(),
                                                      padOp.getResult(), i);
      if (auto v = dyn_cast<Value>(sizes[i]))
        sizes[i] = getAsOpFoldResult(v);
    }

    Value out = rewriter.create<tensor::EmptyOp>(
        paddedOp.getLoc(), sizes, getElementTypeOrSelf(tensorTy));
    auto copied = rewriter.create<linalg::CopyOp>(paddedOp.getLoc(),
                                                  padOp.getResult(), out);
    rewriter.replaceUsesWithIf(padOp.getResult(), copied.getResult(0),
                               [&](OpOperand &opOperand) {
                                 return users.contains(opOperand.getOwner());
                               });
  }

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
  }
}

} // namespace mlir::iree_compiler
