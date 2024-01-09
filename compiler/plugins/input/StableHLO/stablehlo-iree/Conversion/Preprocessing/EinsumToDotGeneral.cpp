// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO einsum op to dot_general ops.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_EINSUMTODOTGENERAL
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {

struct EinsumToDotGeneralPattern final
    : OpRewritePattern<mlir::stablehlo::EinsumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::EinsumOp einsum,
                                PatternRewriter &rewriter) const override {
    StringRef equation = einsum.getEinsumConfig();
    SmallVector<char> lhsTokens, rhsTokens;
    SmallVector<char> resultTokens;
    size_t index = 0;
    enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
    EquationVariable currentVariable = kIsLhs;
    while (index < equation.size()) {
      if (std::isalpha(equation[index])) {
        if (currentVariable == kIsLhs) {
          lhsTokens.push_back(equation[index]);
        } else if (currentVariable == kIsRhs) {
          rhsTokens.push_back(equation[index]);
        } else {
          resultTokens.push_back(equation[index]);
        }
      } else if (equation.substr(index, 1).contains(",")) {
        currentVariable = kIsRhs;
      } else if ((index < (equation.size() - 1)) &&
                 (equation.substr(index, 2).contains("->"))) {
        currentVariable = kIsResult;
        ++index;
      } else {
        return einsum.emitError("unexpected character ")
               << equation.substr(index, 1) << " encountered";
      }
      ++index;
    }

    auto lhsType = cast<RankedTensorType>(einsum.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(einsum.getRhs().getType());
    assert(static_cast<int64_t>(lhsTokens.size()) == lhsType.getRank());
    assert(static_cast<int64_t>(rhsTokens.size()) == rhsType.getRank());

    auto collectOperandDims =
        [resultTokens](
            RankedTensorType operandType, SmallVector<char> operandTokens,
            SmallVector<char> others, SmallVectorImpl<int64_t> &contractingDims,
            SmallVectorImpl<int64_t> &batchingDims,
            SmallVector<char> &dotResultTokens,
            SmallVector<int64_t> &dotResultShape) {
          llvm::SmallDenseSet<char> othersSet(others.begin(), others.end());
          llvm::SmallDenseSet<char> resultTokensSet(resultTokens.begin(),
                                                    resultTokens.end());
          for (auto [idx, token] : llvm::enumerate(operandTokens)) {
            bool isResultToken = resultTokensSet.contains(token);
            bool isOtherToken = othersSet.contains(token);

            if (!isResultToken) {
              contractingDims.push_back(idx);
            } else if (isOtherToken) {
              batchingDims.push_back(idx);
            } else {
              dotResultTokens.push_back(token);
              dotResultShape.push_back(operandType.getShape()[idx]);
            }
          }
        };
    // Indices of batch and contracting dims, relative to each operand's
    // dimensions.
    SmallVector<int64_t> lhsContractingDims, lhsBatchingDims,
        rhsContractingDims, rhsBatchingDims;
    // Tokens representing the natural order of the dot_general op (i.e.
    // the lhs non-contracting followed by rhs non-contracting tokens).
    SmallVector<char> dotResultTokens;
    SmallVector<int64_t> dotResultShape;

    collectOperandDims(lhsType, lhsTokens, rhsTokens, lhsContractingDims,
                       lhsBatchingDims, dotResultTokens, dotResultShape);
    collectOperandDims(rhsType, rhsTokens, lhsTokens, rhsContractingDims,
                       rhsBatchingDims, dotResultTokens, dotResultShape);

    // Prepend batch tokens.
    for (auto [idx, dim] : llvm::enumerate(lhsBatchingDims)) {
      char batchingToken = lhsTokens[dim];
      int64_t batchingShapeDim = lhsType.getShape()[dim];
      dotResultTokens.insert(dotResultTokens.begin() + idx, batchingToken);
      dotResultShape.insert(dotResultShape.begin() + idx, batchingShapeDim);
    }

    // Lowering to dot_general does not support a mismatch between the number
    // of result dims and the number of non-contracting dims.
    if (dotResultTokens.size() != resultTokens.size()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "rank reducing einsum not supported");
    }

    // Generate a permutation sequence based on result tokens.
    SmallVector<int64_t> resultPerms;
    bool isNaturalOrder = true;
    for (char resultToken : resultTokens) {
      auto foundIt = std::find(dotResultTokens.begin(), dotResultTokens.end(),
                               resultToken);
      if (foundIt == dotResultTokens.end()) {
        return rewriter.notifyMatchFailure(
            einsum, "result token not found in operands");
      }
      auto resultIndex = std::distance(dotResultTokens.begin(), foundIt);
      if (resultPerms.empty()) {
        if (resultIndex != 0) {
          isNaturalOrder = false;
        }
      } else if (resultIndex != (resultPerms.back() + 1)) {
        isNaturalOrder = false;
      }
      resultPerms.push_back(resultIndex);
    }

    // Emit the dot_general, using its native result ordering.
    auto dotGeneralResultType = RankedTensorType::get(
        ArrayRef<int64_t>(dotResultShape), lhsType.getElementType());
    auto dimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhsBatchingDims, rhsBatchingDims,
        lhsContractingDims, rhsContractingDims);
    auto dotGeneralOp = rewriter.create<mlir::stablehlo::DotGeneralOp>(
        einsum.getLoc(), dotGeneralResultType, einsum.getLhs(), einsum.getRhs(),
        dimNumbers,
        /*precision_config=*/ArrayAttr{});

    if (isNaturalOrder) {
      // The dot_general is already in an appropriate result order.
      rewriter.replaceOp(einsum, ValueRange{dotGeneralOp});
    } else {
      // Generate a transpose.
      rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
          einsum, dotGeneralOp, rewriter.getDenseI64ArrayAttr(resultPerms));
    }
    return success();
  }
};

struct EinsumToDotGeneral final
    : impl::EinsumToDotGeneralBase<EinsumToDotGeneral> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePreprocessingEinsumToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populatePreprocessingEinsumToDotGeneralPatterns(
    mlir::MLIRContext *context, RewritePatternSet *patterns) {
  patterns->add<EinsumToDotGeneralPattern>(context);
}

} // namespace mlir::iree_compiler::stablehlo
