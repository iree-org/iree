#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "workgroup-calculation"

namespace mlir {
namespace iree_compiler {
namespace utils {
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

/// Computes the bounds of the parallel loops partitioned across workgroups.
static Optional<SmallVector<Value, 2>> getParallelLoopRange(
    PatternRewriter &rewriter, FuncOp numWorkgroupsFn, Location loc,
    linalg::LinalgOp linalgOp) {
  if (!numWorkgroupsFn.empty()) {
    numWorkgroupsFn.emitError("num workgroups fn expected to be empty");
    return {};
  }
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

LogicalResult createNumWorkgroupsFromResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, ArrayRef<int64_t> tileSizes) {
  FuncOp numWorkgroupsFn = getNumWorkgroupsFn(
      linalgOp.getParentOfType<FuncOp>(), numWorkgroupsFnAttr);
  if (!numWorkgroupsFn) return failure();

  Location loc = linalgOp.getLoc();
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, numWorkgroupsFn, loc, linalgOp);
  if (!parallelLoopRange) return failure();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 3> returnValues(3, one);
  for (size_t i = 0, e = std::min<size_t>(parallelLoopRange->size(), 3); i != e;
       ++i) {
    if (tileSizes[e - i - 1] != 0) {
      returnValues[i] = buildCeilDivConstDenominator(
          rewriter, loc, (*parallelLoopRange)[e - i - 1], tileSizes[e - i - 1]);
    }
  }
  rewriter.create<mlir::ReturnOp>(loc, returnValues);
  return success();
}

LogicalResult createNumWorkgroupsFromLinearizedResultShape(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp, FuncOp entryPointFn,
    llvm::StringRef numWorkgroupsFnAttr, int64_t workgroupSizeX) {
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
  OpBuilder::InsertionGuard gaurd(rewriter);
  Optional<SmallVector<Value, 2>> parallelLoopRange =
      getParallelLoopRange(rewriter, numWorkgroupsFn, loc, linalgOp);
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

}  // namespace utils
}  // namespace iree_compiler
}  // namespace mlir
