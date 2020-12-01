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

#include "iree/compiler/Conversion/CodegenUtils/GetNumWorkgroups.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"

#define DEBUG_TYPE "workgroup-calculation"

namespace mlir {
namespace iree_compiler {

FuncOp getNumWorkgroupsFn(FuncOp entryPointFn,
                          llvm::StringRef numWorkgroupsFnAttr) {
  SymbolRefAttr attr =
      entryPointFn.getAttrOfType<SymbolRefAttr>(numWorkgroupsFnAttr);
  if (!attr) {
    entryPointFn.emitError("missing attribute '") << numWorkgroupsFnAttr << "'";
    return nullptr;
  }
  FuncOp numWorkgroupsFn = dyn_cast_or_null<FuncOp>(SymbolTable::lookupSymbolIn(
      entryPointFn.getParentOfType<ModuleOp>(), attr));
  if (!numWorkgroupsFn) {
    entryPointFn.emitError("unable to find num workgroups fn ") << attr;
    return nullptr;
  }
  return numWorkgroupsFn;
}

// TODO: This method is templated on the builder type since the `OpBuilder`
// doesnt have an erase method. Just erasing the op leads to segfaults when the
// builder is `PatternRewriter` since the rewriter doesn't know the op was
// deleted. This can be simplified a lot when this issue is fixed.
template <typename BuilderTy>
static void eraseOp(BuilderTy &builder, Operation *op) {
  builder.eraseOp(op);
}
template <>
void eraseOp(OpBuilder &builder, Operation *op) {
  op->erase();
}

/// Computes the bounds of the loops of the `linalgOp`.
template <typename BuilderTy>
static Optional<SmallVector<Value, 4>> getLoopUpperBounds(
    BuilderTy &builder, Location loc, FuncOp numWorkgroupsFn,
    linalg::LinalgOp linalgOp) {
  if (!numWorkgroupsFn.empty()) {
    numWorkgroupsFn.emitError("num workgroups fn expected to be empty");
    return {};
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Found num workgroups function : "
                 << numWorkgroupsFn.getName();
  });

  builder.createBlock(&numWorkgroupsFn.getBody(), /*insertPt=*/{},
                      numWorkgroupsFn.getType().getInputs());
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(linalgOp, &slice);
  BlockAndValueMapping mapper;
  for (Operation *op : slice) {
    builder.clone(*op, mapper);
  }
  // Clone the linalg operation just to compute the loop bounds.
  linalg::LinalgOp clonedLinalgOp =
      builder.clone(*linalgOp.getOperation(), mapper);
  auto loopRange = clonedLinalgOp.createLoopRanges(builder, loc);
  if (llvm::any_of(loopRange, [](Range range) {
        return !matchPattern(range.stride, m_One()) ||
               !matchPattern(range.offset, m_Zero());
      })) {
    linalgOp.emitError("unhandled non-unit stride loop range");
    return llvm::None;
  }
  SmallVector<Value, 4> bounds = llvm::to_vector<4>(
      llvm::map_range(loopRange, [](Range range) { return range.size; }));
  eraseOp<BuilderTy>(builder, clonedLinalgOp);
  return bounds;
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
static Value buildCeilDiv(OpBuilder &builder, Location loc, Value numerator,
                          Value denominator) {
  Value one = builder.create<ConstantIndexOp>(loc, 1);
  Value t = builder.create<AddIOp>(
      loc, numerator, builder.create<SubIOp>(loc, denominator, one));
  return builder.create<SignedDivIOp>(loc, t, denominator);
}

/// Utility method to build IR that computes ceil(`numerator` / `denominator`)
/// when denominator is a constant.
static Value buildCeilDiv(OpBuilder &builder, Location loc, Value numerator,
                          int64_t denominator) {
  return buildCeilDiv(
      builder, loc, numerator,
      builder.create<ConstantIndexOp>(loc, denominator).getResult());
}

template <class BuilderTy>
static LogicalResult createNumWorkgroupsFromResultShapeImpl(
    BuilderTy &builder, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, ArrayRef<int64_t> tileSizes,
    ArrayRef<unsigned> distributedLoops) {
  FuncOp numWorkgroupsFn = getNumWorkgroupsFn(
      linalgOp.getParentOfType<FuncOp>(), numWorkgroupsFnAttr);
  if (!numWorkgroupsFn) return failure();

  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  auto loopRange = getLoopUpperBounds(builder, loc, numWorkgroupsFn, linalgOp);
  if (!loopRange) return failure();

  SmallVector<Value, 4> numWorkgroups;
  DenseSet<unsigned> distributedLoopsSet(distributedLoops.begin(),
                                         distributedLoops.end());
  for (auto size : enumerate(tileSizes)) {
    if (size.value() && distributedLoopsSet.count(size.index())) {
      Value num =
          buildCeilDiv(builder, loc, (*loopRange)[size.index()], size.value());
      numWorkgroups.push_back(num);
    }
  }
  SmallVector<Value, 4> resultValues =
      llvm::to_vector<4>(llvm::reverse(numWorkgroups));
  Value one = builder.template create<ConstantIndexOp>(loc, 1);
  resultValues.resize(3, one);
  builder.template create<mlir::ReturnOp>(loc, resultValues);
  return success();
}

LogicalResult createNumWorkgroupsFromResultShape(
    OpBuilder &builder, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, ArrayRef<int64_t> tileSizes,
    ArrayRef<unsigned> distributedLoops) {
  return createNumWorkgroupsFromResultShapeImpl<OpBuilder>(
      builder, linalgOp, entryPointFn, numWorkgroupsFnAttr, tileSizes,
      distributedLoops);
}

LogicalResult createNumWorkgroupsFromResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, ArrayRef<int64_t> tileSizes) {
  SmallVector<unsigned, 4> distributedLoops =
      llvm::to_vector<4>(llvm::seq<unsigned>(
          0, std::min<unsigned>(3, getNumOuterParallelLoops(linalgOp))));
  return createNumWorkgroupsFromResultShapeImpl<PatternRewriter>(
      rewriter, linalgOp, entryPointFn, numWorkgroupsFnAttr, tileSizes,
      distributedLoops);
}

LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    ConversionPatternRewriter &rewriter, linalg::LinalgOp linalgOp,
    FuncOp entryPointFn, llvm::StringRef numWorkgroupsFnAttr,
    int64_t workgroupSizeX) {
  FuncOp numWorkgroupsFn = getNumWorkgroupsFn(
      linalgOp.getParentOfType<FuncOp>(), numWorkgroupsFnAttr);
  if (!numWorkgroupsFn) return failure();
  if (!numWorkgroupsFn.empty()) {
    // TODO(ravishankarm): We can end up with multiple linalg operations
    // (typically linalg.generic operations) that have the same workload in a
    // dispatch region. In that case, the first linalg.generic creates the body
    // of number of workgroups. For now, just returning if the body is not empty
    // assuming that it is correct for all the ops in the dispatch region. This
    // needs to be enforced somehow.
    return success();
  }

  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  Optional<SmallVector<Value, 4>> loopRange =
      getLoopUpperBounds(rewriter, loc, numWorkgroupsFn, linalgOp);
  if (!loopRange) return failure();
  unsigned numParallelLoops = getNumOuterParallelLoops(linalgOp);
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (auto range : ArrayRef<Value>(*loopRange).take_front(numParallelLoops)) {
    returnValues[0] = rewriter.create<MulIOp>(loc, range, returnValues[0]);
  }
  returnValues[0] =
      buildCeilDiv(rewriter, loc, returnValues[0], workgroupSizeX);
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

/// The codegeneration emits a function `numWorkgroupsFn` for each entry point
/// function. This function has arguments the !shapex.ranked_shape for all the
/// input and output shaped types. Using this the function returns the number of
/// workgroups to use. To use this function on the host side, generate the
/// !shapex.ranked_shape values that describe the shape of the inputs and
/// outputs of the dispatch region and "inline" the function body.
std::array<Value, 3> calculateWorkgroupCountFromNumWorkgroupsFn(
    Location loc, FuncOp numWorkgroupsFn, IREE::HAL::InterfaceOp interface,
    ArrayRef<Optional<IREE::HAL::TensorRewriteAdaptor>> operands,
    ArrayRef<Optional<IREE::HAL::TensorRewriteAdaptor>> results,
    ConversionPatternRewriter &rewriter) {
  std::array<Value, 3> returnValue = {nullptr, nullptr, nullptr};
  // TODO: This is really just inlining a function. For now assume that the
  // `numWorkgroupsFn` has a single block to make inlining easier.
  if (!numWorkgroupsFn || !llvm::hasSingleElement(numWorkgroupsFn))
    return returnValue;
  SmallVector<SmallVector<Value, 4>, 4> shapeValues;
  shapeValues.reserve(operands.size() + results.size());
  auto getShapeValuesFn =
      [&](ArrayRef<Optional<IREE::HAL::TensorRewriteAdaptor>> values)
      -> LogicalResult {
    for (auto val : values) {
      if (!val) continue;
      Optional<SmallVector<Value, 4>> shape = val->getShapeDims(rewriter);
      if (!shape) return emitError(loc, "shape computation for operand failed");
      shapeValues.push_back(shape.getValue());
    }
    return success();
  };
  if (failed(getShapeValuesFn(operands)) || failed(getShapeValuesFn(results)))
    return returnValue;
  BlockAndValueMapping mapper;
  for (Operation &op : numWorkgroupsFn.front()) {
    if (isa<mlir::ReturnOp>(op)) {
      for (unsigned i = 0, e = std::min<unsigned>(3, op.getNumOperands());
           i != e; ++i) {
        returnValue[i] = mapper.lookupOrNull(op.getOperand(i));
      }
      break;
    }
    if (auto shapeOp = dyn_cast<Shape::RankedDimOp>(op)) {
      if (BlockArgument arg = shapeOp.shape().dyn_cast<BlockArgument>()) {
        auto &dimValues = shapeValues[arg.getArgNumber()];
        mapper.map(shapeOp.result(), dimValues[shapeOp.getIndex()]);
        continue;
      }
      return returnValue;
    }
    // If all its operands are mapped, clone it.
    if (llvm::all_of(op.getOperands(), [&mapper](Value operand) {
          return mapper.contains(operand);
        })) {
      rewriter.clone(op, mapper);
      continue;
    }
  }
  return returnValue;
}

}  // namespace iree_compiler
}  // namespace mlir
