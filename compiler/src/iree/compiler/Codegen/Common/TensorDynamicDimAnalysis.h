// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::iree_compiler {

/// Analysis to compute information about dynamic dimensions of tensors.
///
/// Using the IntegerRangeAnalysis and the IntegerDivisibilityAnalysis
/// this analysis builds information about the range and divisibility of dynamic
/// dimensions of tensor operands in the program. The analysis can then be
/// queried to get the range and divisibility info for any tensor value for any
/// dynamic dimension.
/// TODO: This is not a dataflow analysis or does not update information on IR
/// changes. This could be potentially expensive and is really meant to be used
/// before any transformations to the dispatch. If this needs to be more
/// efficient then this needs to be converted to a data flow solver.
class TensorDynamicDimAnalysis {
public:
  explicit TensorDynamicDimAnalysis(Operation *rootOperation);

  LogicalResult run();

  using TensorDimDivisibilityInfo =
      DenseMap<std::tuple<Value, unsigned>,
               IREE::Util::ConstantIntDivisibility>;
  using TensorDimRangeInfo =
      DenseMap<std::tuple<Value, unsigned>, ConstantIntRanges>;

  std::optional<ConstantIntRanges> getRangeInfo(Value v,
                                                unsigned dimIndex) const {
    auto it = rangeInfo.find({v, dimIndex});
    if (it == rangeInfo.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::optional<IREE::Util::ConstantIntDivisibility>
  getDivisibilityInfo(Value v, unsigned dimIndex) const {
    auto it = divisibilityInfo.find({v, dimIndex});
    if (it == divisibilityInfo.end()) {
      return std::nullopt;
    }
    return it->second;
  }

private:
  DataFlowSolver solver;

  // Operation scope within which the analysis is run.
  Operation *rootOperation;

  // Map of tensor value to integer divisibility information for each dimension.
  TensorDimDivisibilityInfo divisibilityInfo;
  TensorDimRangeInfo rangeInfo;
};

} // namespace mlir::iree_compiler
