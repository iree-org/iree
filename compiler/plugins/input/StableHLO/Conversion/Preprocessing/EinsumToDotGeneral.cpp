// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO einsum op to dot_general ops.

#include <optional>

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_EINSUMTODOTGENERAL
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {

struct EinsumLabels {
  StringRef lhs;
  StringRef rhs;
  StringRef result;
};

// DotGeneral represents each label once per operand. Diagonals and ellipses
// remain einsums for the generic lowering.
static bool hasUniqueEinsumLabels(StringRef labels) {
  llvm::SmallSet<char, 8> seen;
  return llvm::all_of(labels, [&seen](char label) {
    return llvm::isAlpha(label) && seen.insert(label).second;
  });
}

static std::optional<EinsumLabels> getEinsumLabels(StringRef equation) {
  auto arrow = equation.find("->");
  if (arrow == StringRef::npos ||
      equation.drop_front(arrow + 2).contains("->")) {
    return std::nullopt;
  }
  StringRef operands = equation.take_front(arrow);
  if (operands.count(',') != 1) {
    return std::nullopt;
  }
  auto [lhs, rhs] = operands.split(',');
  StringRef result = equation.drop_front(arrow + 2);
  if (!hasUniqueEinsumLabels(lhs) || !hasUniqueEinsumLabels(rhs) ||
      !hasUniqueEinsumLabels(result)) {
    return std::nullopt;
  }
  return EinsumLabels{lhs, rhs, result};
}

struct EinsumDotDimensions {
  SmallVector<int64_t> lhsContracting, rhsContracting;
  SmallVector<int64_t> lhsBatching, rhsBatching;
  SmallVector<char> resultLabels;
};

// Pair shared labels by name, preserving the lhs order on both operands.
// Independent reductions cannot be represented by DotGeneral.
static std::optional<EinsumDotDimensions>
getEinsumDotDimensions(const EinsumLabels &labels) {
  EinsumDotDimensions dimensions;
  SmallVector<char> lhsFreeLabels;
  for (auto [lhsDim, label] : llvm::enumerate(labels.lhs)) {
    size_t rhsDim = labels.rhs.find(label);
    bool inResult = labels.result.contains(label);
    if (rhsDim == StringRef::npos) {
      if (!inResult) {
        return std::nullopt;
      }
      lhsFreeLabels.push_back(label);
    } else if (inResult) {
      dimensions.lhsBatching.push_back(lhsDim);
      dimensions.rhsBatching.push_back(rhsDim);
      dimensions.resultLabels.push_back(label);
    } else {
      dimensions.lhsContracting.push_back(lhsDim);
      dimensions.rhsContracting.push_back(rhsDim);
    }
  }
  llvm::append_range(dimensions.resultLabels, lhsFreeLabels);
  for (char label : labels.rhs) {
    if (labels.lhs.contains(label)) {
      continue;
    }
    if (!labels.result.contains(label)) {
      return std::nullopt;
    }
    dimensions.resultLabels.push_back(label);
  }
  return dimensions;
}

struct EinsumToDotGeneralPattern final
    : OpRewritePattern<mlir::stablehlo::EinsumOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::EinsumOp einsum,
                                PatternRewriter &rewriter) const override {
    auto labels = getEinsumLabels(einsum.getEinsumConfig());
    if (!labels) {
      return rewriter.notifyMatchFailure(einsum,
                                         "requires unique explicit labels");
    }
    auto lhsType = dyn_cast<RankedTensorType>(einsum.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(einsum.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(einsum.getType());
    if (!lhsType || !rhsType || !resultType ||
        static_cast<int64_t>(labels->lhs.size()) != lhsType.getRank() ||
        static_cast<int64_t>(labels->rhs.size()) != rhsType.getRank() ||
        static_cast<int64_t>(labels->result.size()) != resultType.getRank()) {
      return rewriter.notifyMatchFailure(
          einsum, "label counts must match tensor ranks");
    }
    // Bounds are removed before pipeline preprocessing. A standalone rewrite
    // must not permute an encoding without updating its per-axis information.
    if (lhsType.getEncoding() || rhsType.getEncoding() ||
        resultType.getEncoding()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "requires tensors without encodings");
    }
    auto dimensions = getEinsumDotDimensions(*labels);
    if (!dimensions ||
        dimensions->resultLabels.size() != labels->result.size()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "independent reduction unsupported");
    }

    SmallVector<int64_t> resultPerms;
    SmallVector<int64_t> dotResultShape(resultType.getRank());
    for (auto [resultDim, label] : llvm::enumerate(labels->result)) {
      auto position = llvm::find(dimensions->resultLabels, label);
      if (position == dimensions->resultLabels.end()) {
        return rewriter.notifyMatchFailure(
            einsum, "result label not found in operands");
      }
      int64_t dotDim =
          std::distance(dimensions->resultLabels.begin(), position);
      resultPerms.push_back(dotDim);
      // Preserve the declared result precision, including dynamic dimensions.
      dotResultShape[dotDim] = resultType.getDimSize(resultDim);
    }
    auto dotResultType =
        RankedTensorType::get(dotResultShape, resultType.getElementType());
    auto dimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), dimensions->lhsBatching, dimensions->rhsBatching,
        dimensions->lhsContracting, dimensions->rhsContracting);
    auto dotGeneral = mlir::stablehlo::DotGeneralOp::create(
        rewriter, einsum.getLoc(), dotResultType, einsum.getLhs(),
        einsum.getRhs(), dimNumbers, /*precision_config=*/ArrayAttr{},
        mlir::stablehlo::DotAlgorithmAttr{});
    if (llvm::equal(resultPerms, llvm::seq<int64_t>(resultType.getRank()))) {
      rewriter.replaceOp(einsum, dotGeneral.getResult());
      return success();
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
        einsum, dotGeneral, rewriter.getDenseI64ArrayAttr(resultPerms));
    return success();
  }
};

struct EinsumToDotGeneral final
    : impl::EinsumToDotGeneralBase<EinsumToDotGeneral> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePreprocessingEinsumToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
