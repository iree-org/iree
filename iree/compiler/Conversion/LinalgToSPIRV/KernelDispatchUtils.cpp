// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- KernelDispatchUtils.cpp - Utilities for generating dispatch info ---===//
//
// This file defines utility functions that can be used to create information
// the dispatch on the host side needs to execute an entry point function, like
// the number of workgroups to use for launch, etc.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "kernel-dispatch-utils"

namespace mlir {
namespace iree_compiler {

FuncOp getNumWorkgroupsFn(FuncOp entryPointFn) {
  SymbolRefAttr attr =
      entryPointFn.getAttrOfType<SymbolRefAttr>(getNumWorkgroupsFnAttrName());
  if (!attr) {
    entryPointFn.emitError("missing attribute '")
        << getNumWorkgroupsFnAttrName() << "'";
    return nullptr;
  }
  FuncOp numWorkgroupsFn = dyn_cast_or_null<FuncOp>(SymbolTable::lookupSymbolIn(
      entryPointFn.getParentOfType<ModuleOp>(), attr));
  if (!numWorkgroupsFn) {
    entryPointFn.emitError("unable to find num workgroups fn ") << attr;
    return nullptr;
  }
  if (!numWorkgroupsFn.empty()) {
    entryPointFn.emitError("num workgroups fn expected to be empty");
    return nullptr;
  }
  return numWorkgroupsFn;
}

/// Computes the bounds of the parallel loops partitioned across workgroups.
static Optional<SmallVector<Value, 2>> getParallelLoopRange(
    PatternRewriter &rewriter, Location loc, linalg::LinalgOp linalgOp) {
  FuncOp numWorkgroupsFn =
      getNumWorkgroupsFn(linalgOp.getParentOfType<FuncOp>());
  if (!numWorkgroupsFn) return {};
  LLVM_DEBUG({
    llvm::dbgs() << "Found num workgroups function : "
                 << numWorkgroupsFn.getName();
  });
  rewriter.setInsertionPointToEnd(numWorkgroupsFn.addEntryBlock());
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(linalgOp, &slice);
  BlockAndValueMapping mapper;
  for (Operation *op : slice) {
    rewriter.clone(*op, mapper);
  }
  // Clone the linalg operation just to compute the loop bounds.
  linalg::LinalgOp clonedLinalgOp =
      rewriter.clone(*linalgOp.getOperation(), mapper);
  Optional<SmallVector<Value, 4>> bounds =
      getLoopRanges(rewriter, clonedLinalgOp);
  unsigned numParallelLoops = linalgOp.iterator_types()
                                  .getValue()
                                  .take_while([](Attribute attr) -> bool {
                                    return attr.cast<StringAttr>().getValue() ==
                                           getParallelIteratorTypeName();
                                  })
                                  .size();
  SmallVector<Value, 2> returnVals(
      bounds->begin(), std::next(bounds->begin(), numParallelLoops));
  rewriter.eraseOp(clonedLinalgOp);
  return returnVals;
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
static Value buildCeilDiv(PatternRewriter &rewriter, Location loc,
                          Value numerator, Value denominator) {
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  Value t = rewriter.create<AddIOp>(
      loc, numerator, rewriter.create<SubIOp>(loc, denominator, one));
  return rewriter.create<SignedDivIOp>(loc, t, denominator);
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
/// when denominator is a constant.
static Value buildCeilDivConstDenominator(PatternRewriter &rewriter,
                                          Location loc, Value numerator,
                                          int64_t denominator) {
  return buildCeilDiv(rewriter, loc, numerator,
                      rewriter.create<ConstantIndexOp>(loc, denominator));
}

LogicalResult createNumWorkgroupsFromResultShape(PatternRewriter &rewriter,
                                                 linalg::LinalgOp linalgOp,
                                                 FuncOp entryPointFn,
                                                 ArrayRef<int64_t> tileSizes) {
  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, loc, linalgOp);
  if (!parallelLoopRange) return failure();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (size_t i = 0, e = std::min<size_t>(parallelLoopRange->size(), 3); i != e;
       ++i) {
    returnValues[i] = buildCeilDivConstDenominator(
        rewriter, loc, (*parallelLoopRange)[e - i - 1], tileSizes[e - i - 1]);
  }
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    int64_t workgroupSizeX) {
  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, loc, linalgOp);
  if (!parallelLoopRange) return failure();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (auto range : *parallelLoopRange) {
    returnValues[0] = rewriter.create<MulIOp>(loc, range, returnValues[0]);
  }
  returnValues[0] = buildCeilDivConstDenominator(rewriter, loc, returnValues[0],
                                                 workgroupSizeX);
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
