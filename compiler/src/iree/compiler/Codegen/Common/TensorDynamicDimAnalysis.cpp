// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define DEBUG_TYPE "iree-codegen-dynamic-dim-analysis"

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// Helper function to update tensor dynamic dimension info
//===---------------------------------------------------------------------===//

static void
updateRangeInfo(TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo,
                Value v, unsigned dim, const ConstantIntRanges &range) {
  assert(!rangeInfo.contains({v, dim}) &&
         "overwriting existing dim range info");
  rangeInfo.insert({{v, dim},
                    ConstantIntRanges(range.umin(), range.umax(), range.smin(),
                                      range.smax())});
}

static void updateDivisibilityInfo(
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    Value v, unsigned dim,
    const IREE::Util::ConstantIntDivisibility &divisibility) {
  assert(!divisibilityInfo.contains({v, dim}) &&
         "overwriting existing dim divisibility info");
  divisibilityInfo[{v, dim}] = divisibility;
}

// Update the dynamic dim analysis to record the range/divisibility information
// for `tensorValue` at dimension `dimIndex` based on the range/divisibility
// information of an integer/index value `dynamicDim`.
static void updateTensorDimInfo(
    Value tensorValue, unsigned dimIndex, Value dynamicDim,
    const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  // Update range info.
  auto *rangeState =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(dynamicDim);
  if (rangeState && !rangeState->getValue().isUninitialized()) {
    updateRangeInfo(rangeInfo, tensorValue, dimIndex,
                    rangeState->getValue().getValue());
  }

  // Update solver info
  auto *divisibilityState =
      solver.lookupState<IREE::Util::IntegerDivisibilityLattice>(dynamicDim);
  if (divisibilityState && !divisibilityState->getValue().isUninitialized() &&
      divisibilityState->getValue().getValue().sdiv() != 1) {
    updateDivisibilityInfo(divisibilityInfo, tensorValue, dimIndex,
                           divisibilityState->getValue().getValue());
  }
}

//===---------------------------------------------------------------------===//
// Transfer functions for updating dynamic dimension of results of operation.
//===---------------------------------------------------------------------===//

// Helper function to just transfer the range and divisibility information
// `source` value to `dest` value.
static void transferTensorDimInfo(
    Value source, Value dest, const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  // expected that `source` and `dest` are of `RankedTensorType` and of the same
  // type.
  assert(source.getType() == dest.getType());
  auto sourceType = cast<RankedTensorType>(source.getType());
  for (auto index : llvm::seq<unsigned>(0, sourceType.getRank())) {
    // Transfer range info
    auto rangeIt = rangeInfo.find({source, index});
    if (rangeIt != rangeInfo.end()) {
      updateRangeInfo(rangeInfo, dest, index, rangeIt->second);
    }

    auto divisibilityIt = divisibilityInfo.find({source, index});
    if (divisibilityIt != divisibilityInfo.end()) {
      updateDivisibilityInfo(divisibilityInfo, dest, index,
                             divisibilityIt->second);
    }
  }
}

// Update the tensor dimension information for result of a
// `iree_tensor_ext.dispatch.tensor.load` operation.
static void updateTensorDimInfo(
    IREE::TensorExt::DispatchTensorLoadOp flowLoadOp,
    const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  // If there are no dynamic dimensions, nothing to do.
  if (flowLoadOp.getType().hasStaticShape()) {
    return;
  }
  // Check that all strides are 1. Abort otherwise
  if (!llvm::all_of(flowLoadOp.getMixedStrides(), isOneInteger)) {
    return;
  }

  Value result = flowLoadOp.getResult();
  for (auto [index, size] : llvm::enumerate(flowLoadOp.getMixedSizes())) {
    auto dynamicDim = dyn_cast<Value>(size);
    if (!dynamicDim) {
      continue;
    }
    updateTensorDimInfo(result, index, dynamicDim, solver, divisibilityInfo,
                        rangeInfo);
  }
}

// Update the tensor dimension information for result of a `tensor.empty`
// operation.
static void updateTensorDimInfo(
    tensor::EmptyOp emptyOp, const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  auto dimOperands = emptyOp.getOperands();
  if (dimOperands.empty()) {
    return;
  }

  Value result = emptyOp.getResult();
  auto resultType = cast<RankedTensorType>(result.getType());
  int dimOperandIndex = 0;
  for (auto [index, shape] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(shape))
      continue;
    updateTensorDimInfo(result, index, dimOperands[dimOperandIndex++], solver,
                        divisibilityInfo, rangeInfo);
  }
}

// Update the tensor dimension information for results of an operation that
// implements the `DestinationStyleOpInterface`.
static void updateTensorDimInfo(
    DestinationStyleOpInterface dstStyleOp, const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  for (auto [index, result] : llvm::enumerate(dstStyleOp->getResults())) {
    auto resultTensorType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTensorType || resultTensorType.hasStaticShape()) {
      continue;
    }
    Value source = dstStyleOp.getDpsInitOperand(index)->get();
    transferTensorDimInfo(source, result, solver, divisibilityInfo, rangeInfo);
  }
}

// Dispatch to the method that updates the dimension information for an
// operation.
static void updateTensorDimInfo(
    Operation *op, const DataFlowSolver &solver,
    TensorDynamicDimAnalysis::TensorDimDivisibilityInfo &divisibilityInfo,
    TensorDynamicDimAnalysis::TensorDimRangeInfo &rangeInfo) {
  LLVM_DEBUG({
    llvm::dbgs() << "Start updating op\n";
    op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  TypeSwitch<Operation *, void>(op)
      .Case<IREE::TensorExt::DispatchTensorLoadOp, tensor::EmptyOp>(
          [&](auto op) {
            updateTensorDimInfo(op, solver, divisibilityInfo, rangeInfo);
          })
      .Case<DestinationStyleOpInterface>([&](auto op) {
        updateTensorDimInfo(op, solver, divisibilityInfo, rangeInfo);
      });

  LLVM_DEBUG({
    for (auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
      auto tensorType = dyn_cast<RankedTensorType>(result.getType());
      if (!tensorType)
        continue;
      for (auto index : llvm::seq<unsigned>(0, tensorType.getRank())) {
        std::optional<ConstantIntRanges> range;
        std::optional<IREE::Util::ConstantIntDivisibility> divisibility;
        auto rangeIt = rangeInfo.find({result, index});
        if (rangeIt != rangeInfo.end()) {
          range = rangeIt->second;
        }
        auto divisibilityIt = divisibilityInfo.find({result, index});
        if (divisibilityIt != divisibilityInfo.end()) {
          divisibility = divisibilityIt->second;
        }
        if (!range && !divisibility) {
          continue;
        }
        llvm::dbgs() << "\tDim Info: Result number : " << resultIndex
                     << ", dim " << index;
        if (range) {
          llvm::dbgs() << " : Range " << range.value();
        }
        if (divisibility) {
          llvm::dbgs() << " : Divisibility " << divisibility.value();
        }
        llvm::dbgs() << "\n";
      }
    }
  });
}

TensorDynamicDimAnalysis::TensorDynamicDimAnalysis(Operation *rootOp)
    : rootOperation(rootOp) {
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<mlir::dataflow::IntegerRangeAnalysis>();
  solver.load<IREE::Util::IntegerDivisibilityAnalysis>();
}

LogicalResult TensorDynamicDimAnalysis::run() {
  if (failed(solver.initializeAndRun(rootOperation))) {
    return failure();
  }

  // Walk the IR pre-order, forward and update the dynamic information for each
  // tensor.
  rootOperation->walk<WalkOrder::PreOrder>([&](Operation *op) {
    updateTensorDimInfo(op, solver, divisibilityInfo, rangeInfo);
  });

  return success();
}

} // namespace mlir::iree_compiler
