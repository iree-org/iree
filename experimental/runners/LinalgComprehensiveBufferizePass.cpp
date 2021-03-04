//===- LinalgSimpleBufferizePass.cpp - Bufferize Linalg on tensors --------===//
//
// Convert from Linalg ops on tensors to Linalg ops on buffers in a single pass.
// This will aggressively try to perform inplace bufferization and will fail if
// any allocation tries to cross function boundaries or if the pattern
// tensor_load(tensor_memref(x)) is deemed unsafe (very conservative impl for
// now).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "linalg-comprehensive-bufferize-inplace"

using namespace mlir;
using namespace linalg;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace {
struct LinalgComprehensiveBufferizePass
    : public PassWrapper<LinalgComprehensiveBufferizePass,
                         OperationPass<ModuleOp>> {
  LinalgComprehensiveBufferizePass()
      : enablingPassPipeline(OpPassManager("func")) {
    enablingPassPipeline.addPass(createCanonicalizerPass());
    enablingPassPipeline.addPass(createCSEPass());
    enablingPassPipeline.addPass(createLoopInvariantCodeMotionPass());
  }
  LinalgComprehensiveBufferizePass(const LinalgComprehensiveBufferizePass &pass)
      : enablingPassPipeline(pass.enablingPassPipeline) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LinalgDialect, scf::SCFDialect, StandardOpsDialect>();
  }

  void runOnOperation() override;

  void runEnablingTransforms(FuncOp funcOp);
  void bufferizeFuncOpInternals(FuncOp funcOp);

  Option<bool> disableInPlace{
      *this, "disable-inplace",
      llvm::cl::desc(
          "Disables inplace buferization. This is for testing purposes."),
      llvm::cl::init(false)};

  /// Dynamic pass pipeline of transformations that enable better inplace
  /// bufferization.
  OpPassManager enablingPassPipeline;
};
}  // namespace

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify operands that can be bufferized inplace.
constexpr StringLiteral kInPlaceAttrName = "__inplace_attr__";
/// Attribute marker to specify results that fold onto input arguments.
constexpr StringLiteral kResultFoldArgAttrName = "__result_fold_arg_attr__";

// default clause
enum class InPlaceSpec {
  False,
  True,
  None,
};

static StringRef stringify(InPlaceSpec val) {
  switch (val) {
    case InPlaceSpec::False:
      return "false";
    case InPlaceSpec::True:
      return "true";
    case InPlaceSpec::None:
      return "none";
  }
  return "";
}

static Optional<InPlaceSpec> symbolize(StringRef str) {
  return StringSwitch<Optional<InPlaceSpec>>(str)
      .Case("false", InPlaceSpec::False)
      .Case("true", InPlaceSpec::True)
      .Case("none", InPlaceSpec::None)
      .Default(None);
}

/// Set the attribute entry `kInPlaceAttrName`@`idx` to `inplace`.
/// If the attribute does not exist yet, add a blanket array attribute filled
/// with InPlaceSpec::None before setting `kInPlaceAttrName`@`idx` to `inplace`.
static void setInplace(Operation *op, unsigned idx = 0,
                       InPlaceSpec inplace = InPlaceSpec::True) {
  auto attr = op->getAttr(kInPlaceAttrName);
  assert(!attr || attr.isa<ArrayAttr>());
  SmallVector<StringRef> pos;
  if (!attr) {
    auto funcOp = dyn_cast<FuncOp>(op);
    pos = funcOp ? SmallVector<StringRef>(funcOp.getNumArguments(),
                                          stringify(InPlaceSpec::None))
                 : SmallVector<StringRef>(op->getNumOperands(),
                                          stringify(InPlaceSpec::None));
  } else {
    pos = llvm::to_vector<4>(
        attr.cast<ArrayAttr>().getAsValueRange<StringAttr>());
  }
  LLVM_DEBUG(DBGS() << "Set inplace=" << stringify(inplace) << ": " << *op
                    << " @idx=" << idx << "\n");
  pos[idx] = stringify(inplace);
  op->setAttr(kInPlaceAttrName, OpBuilder(op).getStrArrayAttr(pos));
}

static InPlaceSpec getInplace(Operation *op, unsigned operandIndex = 0) {
  auto attr = op->getAttr(kInPlaceAttrName).dyn_cast_or_null<ArrayAttr>();
  if (!attr) return InPlaceSpec::None;
  assert(attr.size() > operandIndex);
  // Must return a proper value.
  return *symbolize(
      *(attr.getAsValueRange<StringAttr>().begin() + operandIndex));
}

static Optional<int64_t> getResultFoldArgIndex(FuncOp op, unsigned resultIdx) {
  auto attr = op->getAttr(kResultFoldArgAttrName).dyn_cast_or_null<ArrayAttr>();
  if (!attr) return llvm::None;
  APInt val = *(attr.getAsValueRange<IntegerAttr>().begin() + resultIdx);
  int64_t res = val.getSExtValue();
  if (res < 0) return llvm::None;
  return res;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific MemRefType support.
//===----------------------------------------------------------------------===//

/// Return the contiguous MemRefType (i.e. with canonical/empty layout map) to
/// which `type` can be bufferized to, assuming `type` is a RankedTensorType.
static MemRefType getContiguousMemRefType(Type type,
                                          ArrayRef<AffineMap> layout = {},
                                          unsigned addressSpace = 0) {
  RankedTensorType tensorType = type.cast<RankedTensorType>();
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, addressSpace);
}

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
static MemRefType getDynamicMemRefType(RankedTensorType tensorType,
                                       unsigned addressSpace = 0) {
  // TODO: address space decisions to connect with the actual alloc.
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, tensorType.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         stridedLayout, addressSpace);
}

// Transfer all `dim` ops on `tensor` to `memref`.
static void transferDimOpsToMemref(Value tensor, Value memref) {
  for (OpOperand &opOperand : llvm::make_early_inc_range(tensor.getUses())) {
    if (isa<DimOp>(opOperand.getOwner())) {
      opOperand.set(memref);
    }
  }
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, ValueRange key, ValueRange value) {
  if (key.empty()) return;
  LLVM_DEBUG(DBGS() << "Map: " << key.front() << " to " << value.front()
                    << "\n");
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, Value key, Value value) {
  LLVM_DEBUG(DBGS() << "Map: " << key << " to " << value << "\n");
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static Value lookup(BlockAndValueMapping &bvm, Value key) {
  if (!bvm.lookupOrNull(key)) {
    MemRefType memRefType =
        getDynamicMemRefType(key.getType().cast<RankedTensorType>());
    Operation *op = key.getDefiningOp() ? key.getDefiningOp()
                                        : key.getParentBlock()->getParentOp();
    OpBuilder b(op->getContext());
    // No InsertionGuard needed here.
    if (auto blockArg = key.dyn_cast<BlockArgument>())
      b.setInsertionPointToStart(blockArg.getParentBlock());
    else
      b.setInsertionPointAfter(op);
    map(bvm, key, b.create<TensorToMemrefOp>(op->getLoc(), memRefType, key));
  }
  return bvm.lookup(key);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific inplace pattern matching support.
//===----------------------------------------------------------------------===//

/// First assign `op` if `slice.back()` isa `T`, then check condition.
/// If anything fails just return failure. Otherwise update `sliceRef` by
/// dropping `sliceRef.back()`, then return success().
template <typename T>
static LogicalResult matchAndDropBack(
    ArrayRef<Operation *> &sliceRef, T &op,
    llvm::function_ref<LogicalResult(T)> condition = nullptr) {
  if (sliceRef.empty()) return failure();
  op = dyn_cast<T>(sliceRef.back());
  if (!op || (condition && failed(condition(op)))) return failure();
  sliceRef = sliceRef.drop_back();
  return success();
}

/// First assign `op1`/`op2` if `slice.front()`/`slice.back()` isa `T1`/`T2`,
/// respectively. Then check condition. If anything fails just return failure.
/// Otherwise update `sliceRef` by dropping `sliceRef.front()` and
/// `sliceRef.back()`, then return success().
template <typename T1, typename T2>
static LogicalResult matchAndDropEnclosingPair(
    ArrayRef<Operation *> &sliceRef, T1 &op1, T2 &op2,
    llvm::function_ref<LogicalResult(T1, T2)> condition = nullptr) {
  if (sliceRef.size() < 2) return failure();
  op1 = dyn_cast<T1>(sliceRef.front());
  op2 = dyn_cast<T2>(sliceRef.back());
  if (!op1 || !op2 || (condition && failed(condition(op1, op2))))
    return failure();
  sliceRef = sliceRef.drop_front().drop_back();
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

// TODO: need to hoist this across function boundaries. Maybe by using
// init_tensor + subtensor_insert.
static Value createNewAllocDeallocPairForShapedValue(
    OpBuilder &b, Location loc, Value shapedValue,
    SmallVector<Value, 4> dynOperands = {}) {
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  assert(memRefType || shapedValue.getType().dyn_cast<RankedTensorType>());
  // TODO: non-zero address space.
  // TODO: layout information if relevant.
  if (!memRefType) memRefType = getContiguousMemRefType(shapedValue.getType());

  OpBuilder::InsertionGuard g(b);
  if (auto bbArg = shapedValue.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    loc = bbArg.getOwner()->getParentOp()->getLoc();
  } else {
    b.setInsertionPointAfter(shapedValue.getDefiningOp());
    loc = shapedValue.getDefiningOp()->getLoc();
  }

  // If the dynOperands are not passed explicity, copmpute them.
  // This circumvents currently missing dim(init_tensor) canonicalizations.
  if (dynOperands.empty()) {
    for (auto dim : llvm::enumerate(memRefType.getShape()))
      if (dim.value() == ShapedType::kDynamicSize)
        dynOperands.push_back(b.create<DimOp>(loc, shapedValue, dim.index()));
  }
  Value allocated = b.create<AllocOp>(loc, memRefType, dynOperands);
  b.setInsertionPoint(allocated.getParentBlock()->getTerminator());
  b.create<DeallocOp>(loc, allocated);
  return allocated;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific inplace analysis support.
//===----------------------------------------------------------------------===//

/// Walk back the chain of known ops all the way to function arguments:
///   - if an AllocOp, AllocaOp or InitTensorOp is met, return true.
///   - if a LinalgOp is met, return true: either it is already known to trace
///     back to a function arg that is writeable or it is already guaranteed to
///     create an AllocOp into which we can write.
///   - if the function argument is marked inplace, return true.
///   - if the function argument is not marked inplace, return false.
///   - if an unknown op is encountered, abort for now.
static bool livesInWritableMemoryLocation(Value v) {
  LLVM_DEBUG(DBGS() << "Start livesInWritableMemoryLocation @" << v << "\n");
  bool done = false, res = false;
  while (!done) {
    // Scalar or vector value comes from a load, just return true.
    if (!v.getType()
             .isa<MemRefType, RankedTensorType, UnrankedMemRefType,
                  UnrankedTensorType>())
      return true;
    if (auto bbArg = v.dyn_cast<BlockArgument>()) {
      llvm::TypeSwitch<Operation *, void>(bbArg.getOwner()->getParentOp())
          .Case([&](scf::ForOp forOp) {
            v = forOp.getIterOperands()[bbArg.getArgNumber() - /*iv=*/1];
          })
          .Case([&](FuncOp funcOp) {
            assert(bbArg.getType().isa<TensorType>() &&
                   "already bufferized func");
            if (getInplace(funcOp, bbArg.getArgNumber()) != InPlaceSpec::True)
              res = false;
            else
              res = true;
            done = true;
          })
          .Default([&](Operation *op) {
            llvm::errs() << "In function:\n" << *op->getParentOfType<FuncOp>();
            llvm::errs() << "\nUnsupported livesInWritableMemoryLocation "
                         << *op << "\nstarting from value: " << v;
            abort();
          });
      continue;
    }
    auto opResult = v.cast<OpResult>();
    llvm::TypeSwitch<Operation *, void>(opResult.getOwner())
        .Case([&](LinalgOp linalgOp) {
          // TODO: uses implicit knowledge that output tensor matches result
          // 1-1.
          v = linalgOp.getOutputTensors()[opResult.getResultNumber()];
        })
        .Case<TensorToMemrefOp, TensorLoadOp, tensor::CastOp>(
            [&](Operation *op) { v = op->getOperand(0); })
        .Case<linalg::InitTensorOp, AllocOp, AllocaOp>([&](Operation *op) {
          res = true;
          done = true;
        })
        .Default([&](Operation *op) {
          llvm::errs() << "In function:\n" << *op->getParentOfType<FuncOp>();
          llvm::errs() << "\nUnsupported livesInWritableMemoryLocation " << *op
                       << "\nstarting from value: " << v;
          abort();
        });
  }
  return res;
}

namespace {
// Represent an inplace action that is to be committed as an Operation attribute
// upon successful detection of a hain of ops that can be run inplace.
struct InPlaceAction {
  Operation *op;
  SmallVector<unsigned> outputIndices;
};
}  // namespace

/// Find simple forms of destructive update which writes over a yielded tensor
/// without ever reading from it. For now, we only allow:
/// ```
///    vector.transfer_write -> subtensor_insert -> yield
/// ```
static void iterativeOverwritesAnalysis(Operation *parentOp,
                                        ArrayRef<BlockArgument> candidates) {
  if (!isa<scf::ForOp, FuncOp>(parentOp)) return;

  for (auto en : llvm::enumerate(candidates)) {
    Value candidate = en.value();
    if (!candidate.getType().isa<ShapedType>()) continue;

    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(DBGS() << "Iterative overwrite analysis on candidate: "
                      << candidate << "\nof:\n"
                      << *parentOp << "\n");
    if (!livesInWritableMemoryLocation(candidate)) continue;

    llvm::SetVector<Operation *> slice;
    getForwardSlice(candidate, &slice, [&](Operation *op) {
      // Skip any extra nesting between parentOp and op.
      return op == parentOp || op->getBlock()->getParentOp() == parentOp;
    });

    LLVM_DEBUG(DBGS() << "Iterative overwrite TRY:\n");
    LLVM_DEBUG(llvm::for_each(
        slice, [](Operation *op) { DBGS() << "Slice op: " << *op << "\n"; }));

    // bbArg must be used exactly by one subtensor_insert + yield.
    if (!candidate.hasOneUse()) {
      LLVM_DEBUG(DBGS() << "bbArg does not have exactly 1 use."
                           "\nIterative overwrite FAIL\n");
      continue;
    }
    if (slice.size() != 2) {
      LLVM_DEBUG(DBGS() << "Need exactly 2 ops in slice. "
                           "\nIterative overwrite FAIL\n");
      continue;
    }

    auto sliceRef = slice.getArrayRef();
    // Match yieldOp and update sliceRef.
    scf::YieldOp yieldOp;
    if (failed(matchAndDropBack(sliceRef, yieldOp))) continue;

    // Match subTensorInsertOp and update sliceRef.
    SubTensorInsertOp subTensorInsertOp;
    if (failed(matchAndDropBack(sliceRef, subTensorInsertOp))) continue;

    // Optional vector::TransferWriteOp.
    auto vectorTransferWriteOp =
        subTensorInsertOp.source().getDefiningOp<vector::TransferWriteOp>();

    // subtensor_insert must be used exactly by the yield at index `idx`.
    unsigned idx = en.index();
    if (!subTensorInsertOp.result().hasOneUse() ||
        !isa<scf::YieldOp>(*subTensorInsertOp.result().getUsers().begin()) ||
        subTensorInsertOp.result().getUses().begin()->getOperandNumber() !=
            idx) {
      LLVM_DEBUG(DBGS() << "SubTensorInsertOp does not have a single YieldOp "
                           "use. \nIterative overwrite chain FAIL\n");
      continue;
    }

    setInplace(parentOp, en.index());
    if (vectorTransferWriteOp) setInplace(vectorTransferWriteOp);
    setInplace(subTensorInsertOp);
    setInplace(yieldOp, en.index());
    LLVM_DEBUG(DBGS() << "Iterative overwrite chain SUCCESS\n");
  }
}

/// Return true is all offsets, sizes and strides are equal.
static LogicalResult sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface op1, OffsetSizeAndStrideOpInterface op2) {
  if (op1.static_offsets().size() != op2.static_offsets().size())
    return failure();
  if (op1.static_sizes().size() != op2.static_sizes().size()) return failure();
  if (op1.static_strides().size() != op2.static_strides().size())
    return failure();
  for (auto it : llvm::zip(op1.getMixedOffsets(), op2.getMixedOffsets()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  for (auto it : llvm::zip(op1.getMixedSizes(), op2.getMixedSizes()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  for (auto it : llvm::zip(op1.getMixedStrides(), op2.getMixedStrides()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  return success();
}

static LogicalResult matchingVectorTransfersAtSource(
    vector::TransferReadOp read, vector::TransferWriteOp write,
    Value subtensor) {
  // Either we have a pair of matching transfer read/write or none.
  if (read && !write) {
    LLVM_DEBUG(DBGS() << "Slice has transferReadOp but no transferWriteOp"
                         "\nDestructive update chain FAIL\n");
    return failure();
  }
  if (!read && write) {
    LLVM_DEBUG(DBGS() << "Slice has transferWriteOp but no transferReadOp"
                         "\nDestructive update chain FAIL\n");
    return failure();
  }
  if (read && write) {
    // If we have a pair of mathing read/write, the tensor and vector shape
    // must exactly match (i.e. this is a vectorization).
    if (read.source() != subtensor) {
      LLVM_DEBUG(DBGS() << "transferReadOp.source() != subTensor.result()"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (write.source() != subtensor) {
      LLVM_DEBUG(DBGS() << "transferWriteOp.source() != subTensor.result()"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (read.getShapedType().getShape() != read.getVectorType().getShape()) {
      LLVM_DEBUG(DBGS() << "transferReadOp source and result shapes mismatch"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (write.getShapedType().getShape() != write.getVectorType().getShape()) {
      LLVM_DEBUG(DBGS() << "transferWriteOp source and result shapes mismatch"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
  }
  return success();
}

/// In the case of an scf::ForOp, we look for:
///   `candidate -> subtensor -> vector.transfer_read(*) -> ...
///      vector.transfer_write(*) -> subtensor_insert -> return`.
/// sliceRef is automaticaly updated to match `...`.
///
/// (*) represents an optional op in the chain, if a subtensor or
/// vector.transfer is included, the matching op must be included too.
static LogicalResult detectDestructiveUpdatePattern(
    FuncOp parentOp, BlockArgument candidate, ArrayRef<Operation *> &sliceRef,
    SmallVector<InPlaceAction> &inPlaceActions) {
  if (!parentOp) return failure();

  ReturnOp terminator;
  // Match returnOp and update sliceRef.
  if (failed(matchAndDropBack(sliceRef, terminator))) {
    LLVM_DEBUG(DBGS() << "destructive update slice must end with a known "
                         "terminator.\nDestructive update chain FAIL\n");
    return failure();
  }
  return success();
}

/// In the case of an scf::ForOp, we look for:
///   `candidate -> subtensor -> vector.transfer_read(*) -> ...
///      vector.transfer_write(*) -> subtensor_insert -> yield`.
/// sliceRef is automaticaly updated to match `...`.
///
/// (*) represents an optional op in the chain, if a subtensor or
/// vector.transfer is included, the matching op must be included too.
static LogicalResult detectDestructiveUpdatePattern(
    scf::ForOp parentOp, BlockArgument candidate,
    ArrayRef<Operation *> &sliceRef,
    SmallVector<InPlaceAction> &inPlaceActions) {
  if (!parentOp) return failure();

  scf::YieldOp terminator;
  SubTensorOp subTensorOp;
  SubTensorInsertOp subTensorInsertOp;
  vector::TransferReadOp vectorTransferReadOp;
  vector::TransferWriteOp vectorTransferWriteOp;

  // bbArg must be used exactly by one subtensor / subtensor_insert pair.
  if (candidate.use_empty() || candidate.hasOneUse() ||
      std::next(candidate.getUsers().begin(), 2) !=
          candidate.getUsers().end()) {
    LLVM_DEBUG(DBGS() << "bbArg does not have exactly 2 uses."
                         "\nDestructive update chain FAIL\n");
    return failure();
  }
  if (sliceRef.size() < 3) {
    LLVM_DEBUG(DBGS() << "scf::ForOp destructive updated must have >= 3 ops."
                         "\nDestructive update chain FAIL\n");
    return failure();
  }

  // Match yieldOp and update sliceRef.
  if (failed(matchAndDropBack(sliceRef, terminator))) {
    LLVM_DEBUG(DBGS() << "destructive update slice must end with a known "
                         "terminator.\nDestructive update chain FAIL\n");
    return failure();
  }

  // Match subtensor pair and update sliceRef.
  // subtensor / subtensor_insert must match.
  auto matchSubTensors = [](SubTensorOp st, SubTensorInsertOp sti) {
    auto res = sameOffsetsSizesAndStrides(st, sti);
    if (failed(res))
      LLVM_DEBUG(DBGS() << "subtensor ops don't match: " << st << " and " << sti
                        << "\nDestructive update chain FAIL\n");
    return res;
  };
  if (failed(matchAndDropEnclosingPair<SubTensorOp, SubTensorInsertOp>(
          sliceRef, subTensorOp, subTensorInsertOp, matchSubTensors)))
    return failure();

  // subtensor_insert must be used exactly by the terminator at index `idx`.
  unsigned idx = candidate.getArgNumber() - /*#iv=*/1;  // adjust for ForOp iv.
  if (!subTensorInsertOp.result().hasOneUse() ||
      terminator != *subTensorInsertOp.result().getUsers().begin() ||
      terminator->getOperand(idx) != subTensorInsertOp.result()) {
    LLVM_DEBUG(
        DBGS() << "SubTensorInsertOp does not have a single terminator use "
                  "at the right index.\nDestructive update chain FAIL\n");
    return failure();
  }

  // Maybe match vector transfer pair and update sliceRef.
  // If we find one, the other must be present and match too.
  auto matchTransfers = [&](vector::TransferReadOp read,
                            vector::TransferWriteOp write) {
    return matchingVectorTransfersAtSource(read, write, subTensorOp.result());
  };
  if (failed(matchAndDropEnclosingPair<vector::TransferReadOp,
                                       vector::TransferWriteOp>(
          sliceRef, vectorTransferReadOp, vectorTransferWriteOp,
          matchTransfers)) &&
      (vectorTransferReadOp || vectorTransferWriteOp))
    return failure();

  // Commit what has been detected.
  inPlaceActions.push_back(InPlaceAction{subTensorOp});
  if (vectorTransferReadOp)
    inPlaceActions.push_back(InPlaceAction{vectorTransferReadOp});
  if (vectorTransferWriteOp)
    inPlaceActions.push_back(InPlaceAction{vectorTransferWriteOp});
  inPlaceActions.push_back(InPlaceAction{subTensorInsertOp});
  inPlaceActions.push_back(InPlaceAction{terminator, {idx}});

  return success();
}

/// Iterate over bbArgs of `parentOp` and determine if they are the root of a
/// destructive update chain such as:
/// ```
///    scf.for bbArg -> subtensor -> DAG of admissible inPlaceActions
///      -> subtensor_insert -> yield.
/// ```
/// Such a representation is related to traditional loop nest + memory analysis
/// but provides a simpler abstraction.
/// In traditional memory-based dependence analysis, one would need to analyze
/// all possible interleavings of possibly aliasing loads and stores in the
/// context of the k-common surrounding loops.
/// With scf.for + subtensor + subtensor_insert + yield, more ordering semantics
/// are available as well as dealiasing thanks to SSA use-def chains.
static void destructiveUpdateAnalysis(Operation *parentOp,
                                      ArrayRef<BlockArgument> candidates) {
  for (auto en : llvm::enumerate(candidates)) {
    BlockArgument candidate = en.value();
    if (!candidate.getType().isa<ShapedType>()) continue;

    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(DBGS() << "Destructive update analysis on candidate: "
                      << candidate << "\nof:\n"
                      << *parentOp << "\n");
    if (!livesInWritableMemoryLocation(candidate)) continue;

    llvm::SetVector<Operation *> slice;
    getForwardSlice(candidate, &slice, [&](Operation *op) {
      // Skip any extra nesting between parentOp and op.
      return op == parentOp || op->getBlock()->getParentOp() == parentOp;
    });

    LLVM_DEBUG(DBGS() << "Slice:\n");
    for (auto *op : slice) LLVM_DEBUG(DBGS() << *op << "\n");

    SmallVector<InPlaceAction> inPlaceActions;
    inPlaceActions.reserve(slice.size());
    ArrayRef<Operation *> sliceRef = slice.getArrayRef();
    if (failed(detectDestructiveUpdatePattern(dyn_cast<scf::ForOp>(parentOp),
                                              candidate, sliceRef,
                                              inPlaceActions)) &&
        failed(detectDestructiveUpdatePattern(
            dyn_cast<FuncOp>(parentOp), candidate, sliceRef, inPlaceActions))) {
      LLVM_DEBUG(DBGS() << "Failed to detect: Destructive update chain FAIL\n");
      continue;
    }

    // Add the current op and add pattern eagerly to simplify implementation.
    inPlaceActions.push_back(
        {parentOp, {static_cast<unsigned int>(en.index())}});
    for (auto &action : inPlaceActions) {
      if (action.outputIndices.empty()) setInplace(action.op);
      for (unsigned idx : action.outputIndices) setInplace(action.op, idx);
    }
  }

  parentOp->walk([](Operation *op) {
    if (isa<TensorLoadOp, TensorToMemrefOp>(op)) setInplace(op);
    if (auto linalgOp = dyn_cast<LinalgOp>(op)) {
      // For now, just check that the operand and corresponding result have
      // 0 uses. In the future we can build a cost-model to take care of
      // diamond dependences.
      unsigned resultIdx = 0;
      for (auto &opOperand : linalgOp.getOutputTensorsOpOperands()) {
        if (opOperand->get().hasOneUse() &&
            linalgOp->getResult(resultIdx).hasOneUse())
          setInplace(op, opOperand->getOperandNumber());
        ++resultIdx;
      }
    }
  });
}

static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym) return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

static void inplaceAnalysisFuncOpInternals(FuncOp funcOp) {
  funcOp.walk([&](scf::ForOp forOp) {
    iterativeOverwritesAnalysis(forOp, forOp.getRegionIterArgs());
  });
  iterativeOverwritesAnalysis(funcOp, funcOp.getArguments());
  funcOp.walk([&](scf::ForOp forOp) {
    destructiveUpdateAnalysis(forOp, forOp.getRegionIterArgs());
  });
  destructiveUpdateAnalysis(funcOp, funcOp.getArguments());
}

/// Analyse a `callOp` to a FuncOp and determine whether any of its tensor
/// operand could be safely written inplace after it is converted to buffer
/// form by a bufferization process. Iterate on the uses of callOp's operands
/// to determine whether all such uses dominate callOp. If any use of an
/// operand does not dominate `callOp`, this means that the operand tensor
/// value may be needed somewhere else and it is illegal to update in-place
/// after bufferization. Add a `kInPlaceAttrName` string attribute to `callOp`
/// to carry the result of this analysis until bufferization is completed. The
/// "meet" of all `kInPlaceAttrName` for all `callOp` to a given FuncOp
/// determines the `kInPlaceAttrName` for that FuncOp.
static void inplaceFunctionArgumentAnalysis(CallOpInterface callOp,
                                            DominanceInfo &domInfo) {
  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp) return;

  if (llvm::none_of(callOp->getOperandTypes(),
                    [](Type t) { return t.isa<TensorType>(); }))
    return;

  LLVM_DEBUG(DBGS() << "Begin inplaceFunctionArgumentAnalysis within:\n"
                    << *callOp->getParentOfType<FuncOp>()
                    << "callOp: " << *callOp << "\n";);
  for (OpOperand &opOperand : callOp->getOpOperands()) {
    Value tensor = opOperand.get();
    if (!tensor.getType().isa<TensorType>()) continue;

    unsigned idx = opOperand.getOperandNumber();
    LLVM_DEBUG(DBGS() << "tensor @idx=" << idx << ": " << tensor << "\n");

    // For now, assume any use is a read.
    // Write-only is a non-problem: will represent with shapes in the future.
    // If any use of the tensor does not properly dominate callOp, we can't
    // bufferize the tensor inplace.
    InPlaceSpec callInPlace = InPlaceSpec::True;
    for (auto &use : tensor.getUses()) {
      Operation *user = use.getOwner();
      if (domInfo.properlyDominates(user, callOp)) continue;
      if (use.getOperandNumber() == idx) continue;
      LLVM_DEBUG(DBGS() << "non-properly dominate user: " << *user << "\n");
      callInPlace = InPlaceSpec::False;
      break;
    }
    // CallOp instance can immediately determine whether it allows inplace.
    setInplace(callOp, idx, callInPlace);
    // FuncOp inplace is the meet of all the calls.
    InPlaceSpec funcInPlace = getInplace(funcOp, idx);
    if (funcInPlace == InPlaceSpec::False) continue;
    setInplace(funcOp, idx, callInPlace);
  }

  LLVM_DEBUG(DBGS() << "End inplaceFunctionArgumentAnalysis within:\n"
                    << *callOp->getParentOfType<FuncOp>()
                    << "callOp: " << *callOp << "\n";);
}

//===----------------------------------------------------------------------===//
// Bufferization as simple BlockAndValueMapping rewrites / without
// conversions.
//===----------------------------------------------------------------------===//

/// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
/// This works on mixed tensor + buffer Linalg ops: some results may have been
/// already bufferized by a previous destructive update bufferization.
/// Allocate the output buffers for the remaining tensor output operands of
/// the Linalg op. If the tensor is an "init" tensor (i.e. its value is
/// actually used in the payload region), we additionally copy the original
/// value into the newly allocated buffer.
static LogicalResult allocateBuffersForResults(
    OpBuilder &b, Location loc, LinalgOp op,
    SmallVectorImpl<Value> &resultBuffers, BlockAndValueMapping &bvm) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Linalg invariant: output tensors and result match 1-1.
  assert(op.getNumOutputTensors() == op->getNumResults());
  for (auto &opOperand : op.getOutputOpOperands()) {
    Value output = opOperand.get();
    if (output.getType().isa<MemRefType>()) {
      resultBuffers.push_back(output);
      continue;
    }

    // If output tensor is marked inplace, just use the buffer.
    if (getInplace(op, opOperand.getOperandNumber()) == InPlaceSpec::True) {
      resultBuffers.push_back(lookup(bvm, output));
      continue;
    }

    Value dimTensor = bvm.lookupOrDefault(output);
    Value alloc = createNewAllocDeallocPairForShapedValue(b, loc, dimTensor);
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOpOperand(&opOperand))
      b.create<CopyOp>(loc, lookup(bvm, output), alloc);
  }
  map(bvm, op->getResults(), resultBuffers);
  for (auto it : llvm::zip(op->getResults(), resultBuffers)) {
    transferDimOpsToMemref(std::get<0>(it), std::get<1>(it));
  }
  return success();
}

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
static void finalizeBufferAllocation(OpBuilder &b, LinalgOp op,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = op.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  Location loc = op.getLoc();
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  map(bvm, op.getOperation()->getResults(), outputs);
  for (auto it : llvm::zip(op.getOperation()->getResults(), outputs)) {
    transferDimOpsToMemref(std::get<0>(it), std::get<1>(it));
  }

  if (!op.hasTensorSemantics()) op->erase();
}

/// Generic conversion pattern that matches any LinalgOp. This avoids
/// template instantiating one pattern for each LinalgOp.
/// This works on mixed tensor + buffer Linalg ops: some results may have been
/// already bufferized by a previousdestructive update bufferization.
static LogicalResult convertAnyLinalgOp(OpBuilder &b, LinalgOp op,
                                        BlockAndValueMapping &bvm) {
  if (op.hasBufferSemantics()) return failure();

  LLVM_DEBUG(DBGS() << "convertAnyLinalgOp: " << *op << "\n");

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (Value v : op.getInputs()) {
    newInputBuffers.push_back(lookup(bvm, v));
  }
  SmallVector<Value, 2> newOutputBuffers;
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm)))
    assert(false);

  // Delegate to the linalg generic pattern.
  if (auto genericOp = dyn_cast<GenericOp>(op.getOperation())) {
    finalizeBufferAllocation(b, genericOp, newInputBuffers, newOutputBuffers,
                             bvm);
    return success();
  }

  SmallVector<Value, 2> newResults;
  for (OpOperand &outputOpOperand : op.getOutputOpOperands()) {
    Value output = outputOpOperand.get();
    if (output.getType().isa<MemRefType>()) continue;
    auto tensorType = output.getType().cast<RankedTensorType>();
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(op);
    Value tensor = b.create<TensorLoadOp>(
        loc, tensorType,
        newOutputBuffers[outputOpOperand.getOperandNumber() -
                         op.getNumInputs()]);
    newResults.push_back(tensor);
    map(bvm, tensor,
        newOutputBuffers[outputOpOperand.getOperandNumber() -
                         op.getNumInputs()]);
  }
  // Can't just map.
  // map(bvm, op.getOutputs(), newOutputBuffers);
  // map(bvm, op->getResults(), newResults);
  // Must explicitly push value out because conume ops are not guaranteed to
  // pull the value from bvm (e.g. scf.for with core bufferization use
  // conversion patterns).
  op->replaceAllUsesWith(newResults);

  finalizeBufferAllocation(b, op, newInputBuffers, newOutputBuffers, bvm);

  return success();
}

static LogicalResult convertTransferOp(OpBuilder &b,
                                       VectorTransferOpInterface op,
                                       BlockAndValueMapping &bvm) {
  if (op.getShapedType().isa<MemRefType>()) return failure();

  assert(op->getNumResults() == 1);
  Value outputTensor = op->getResult(0);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value newInputBuffer = lookup(bvm, op.source());
  if (auto tensorType =
          op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
    Value tensor = bvm.lookupOrDefault(outputTensor);
    Value alloc = createNewAllocDeallocPairForShapedValue(b, loc, tensor);
    map(bvm, op->getResult(0), alloc);
    transferDimOpsToMemref(op->getResult(0), alloc);
  }

  // Replace the tensor operand.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    readOp.sourceMutable().assign(newInputBuffer);
  } else {
    auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());
    // Create a new transfer_write on buffer that doesn't have a return value.
    // Leave the previous transfer_write to dead code as it still has uses at
    // this point.
    b.create<vector::TransferWriteOp>(
        loc, writeOp.vector(), newInputBuffer, writeOp.indices(),
        writeOp.permutation_map(),
        writeOp.masked() ? *writeOp.masked() : ArrayAttr());

    Value tensor = b.create<TensorLoadOp>(
        loc, writeOp.getResult(0).getType().cast<RankedTensorType>(),
        newInputBuffer);
    SmallVector<Value, 1> newResult(1, {tensor});
    writeOp.replaceAllUsesWith(newResult);
    map(bvm, tensor, newInputBuffer);
  }
  return success();
}

static LogicalResult convertInitTensorOp(OpBuilder &b,
                                         InitTensorOp initTensorOp,
                                         BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(initTensorOp);
  Value alloc = createNewAllocDeallocPairForShapedValue(
      b, initTensorOp->getLoc(), initTensorOp.result(), initTensorOp.sizes());
  map(bvm, initTensorOp.result(), alloc);
  return success();
}

static LogicalResult convertPadTensorOp(OpBuilder &b, PadTensorOp padTensorOp,
                                        BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(padTensorOp);
  auto tensorType = padTensorOp.result().getType().cast<RankedTensorType>();
  auto sourceMemRef = lookup(bvm, padTensorOp.source());
  auto sourceMemRefType = sourceMemRef.getType().cast<MemRefType>();
  auto memRefType =
      getContiguousMemRefType(tensorType, sourceMemRefType.getAffineMaps(),
                              sourceMemRefType.getMemorySpace());
  Value res =
      b.create<MemRefCastOp>(padTensorOp.getLoc(), memRefType, sourceMemRef);
  map(bvm, padTensorOp.result(), res);
  return success();
}

static LogicalResult convertSubTensorInsertOp(
    OpBuilder &b, SubTensorInsertOp subTensorInsertOp,
    BlockAndValueMapping &bvm) {
  Location loc = subTensorInsertOp.getLoc();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensorInsertOp);
  Value dstMemref = lookup(bvm, subTensorInsertOp.dest());
  auto dstMemrefType = dstMemref.getType().cast<MemRefType>();
  Value srcMemref = lookup(bvm, subTensorInsertOp.source());
  auto subviewMemRefType =
      SubViewOp::inferRankReducedResultType(
          subTensorInsertOp.getSourceType().getRank(), dstMemrefType,
          subTensorInsertOp.getMixedOffsets(),
          subTensorInsertOp.getMixedSizes(),
          subTensorInsertOp.getMixedStrides())
          .cast<MemRefType>();
  // Take a subview of the dst.
  Value subView = b.create<SubViewOp>(
      loc, subviewMemRefType, dstMemref, subTensorInsertOp.getMixedOffsets(),
      subTensorInsertOp.getMixedSizes(), subTensorInsertOp.getMixedStrides());
  // Linalg op and vector.transfer_write producers directly write their output
  // buffer. If the producer is not one of these ops or if it subtensor_insert
  // is not marked inplace, we ened to copy.
  bool isInPlaceProducer =
      subTensorInsertOp.source().getDefiningOp<LinalgOp>() ||
      subTensorInsertOp.source().getDefiningOp<vector::TransferWriteOp>();
  if (!isInPlaceProducer || getInplace(subTensorInsertOp) != InPlaceSpec::True)
    b.create<CopyOp>(subTensorInsertOp.getLoc(), srcMemref, subView);
  Value tensor = b.create<TensorLoadOp>(
      loc, subTensorInsertOp->getResult(0).getType(), dstMemref);
  SmallVector<Value, 1> newResult(1, {tensor});
  subTensorInsertOp->replaceAllUsesWith(newResult);
  map(bvm, tensor, dstMemref);
  return success();
}

static LogicalResult convertSubTensorOp(OpBuilder &b, SubTensorOp subTensor,
                                        BlockAndValueMapping &bvm) {
  Location loc = subTensor.getLoc();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensor);
  Value srcMemref = lookup(bvm, subTensor.source());
  auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
  auto dstTensorType = subTensor.result().getType().cast<RankedTensorType>();

  auto subviewMemRefType =
      SubViewOp::inferRankReducedResultType(
          dstTensorType.getRank(), srcMemrefType, subTensor.getMixedOffsets(),
          subTensor.getMixedSizes(), subTensor.getMixedStrides())
          .cast<MemRefType>();

  Value subView = b.create<SubViewOp>(
      loc, subviewMemRefType, srcMemref, subTensor.getMixedOffsets(),
      subTensor.getMixedSizes(), subTensor.getMixedStrides());
  map(bvm, subTensor.result(), subView);
  return success();
}

static LogicalResult convertTensorCastOp(OpBuilder &b, tensor::CastOp castOp,
                                         BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(castOp);
  auto sourceMemRefType =
      lookup(bvm, castOp.source()).getType().dyn_cast<MemRefType>();
  Type memRefType;
  TensorType tensorType = castOp.getResult().getType().cast<TensorType>();
  if (tensorType.isa<UnrankedTensorType>()) {
    memRefType = UnrankedMemRefType::get(tensorType.getElementType(),
                                         sourceMemRefType.getMemorySpace());
  } else {
    memRefType =
        getContiguousMemRefType(tensorType, sourceMemRefType.getAffineMaps(),
                                sourceMemRefType.getMemorySpace());
  }
  Value res = b.create<MemRefCastOp>(castOp.getLoc(), memRefType,
                                     lookup(bvm, castOp.source()));
  map(bvm, castOp.getResult(), res);
  return success();
}

static void bufferizeFunctionCallBoundaries(FuncOp funcOp) {
  // kResultFoldArgAttrName is set once funcOp is bufferized.
  if (funcOp->getAttr(kResultFoldArgAttrName)) return;

  SmallVector<int64_t> resultArgumentFolding(
      funcOp.type().cast<FunctionType>().getNumResults(), -1);

  LLVM_DEBUG(DBGS() << "Begin bufferizeFunctionCallBoundaries:\n" << funcOp);

  // Take the terminator (assume the last block is the only one that has it).
  auto returnOp = cast<ReturnOp>(funcOp.body().back().getTerminator());
  for (OpOperand &returnOpOperand : returnOp->getOpOperands()) {
    Value returnValue = returnOpOperand.get();
    unsigned returnIndex = returnOpOperand.getOperandNumber();
    if (!returnValue.getType().isa<RankedTensorType>()) continue;

    // If returned value is a bbArg, it only folds if it is a function
    // argument.
    BlockArgument bbArg = returnValue.dyn_cast<BlockArgument>();
    if (bbArg) {
      if (returnValue == funcOp.getArgument(bbArg.getArgNumber()))
        resultArgumentFolding[returnIndex] = bbArg.getArgNumber();
      else
        continue;
    }

    // Otherwise we look for tensor_load(tensor_to_memref(bbarg)).
    auto tensorLoadOp = returnValue.getDefiningOp<TensorLoadOp>();
    if (!tensorLoadOp) continue;
    auto tensorToMemRefOp =
        tensorLoadOp.memref().getDefiningOp<TensorToMemrefOp>();
    if (!tensorToMemRefOp) continue;

    // If returned value is a bbArg, it only folds if it is a function
    // argument.
    bbArg = tensorToMemRefOp.tensor().dyn_cast<BlockArgument>();
    if (bbArg) {
      if (bbArg == funcOp.getArgument(bbArg.getArgNumber()))
        resultArgumentFolding[returnIndex] = bbArg.getArgNumber();
      else
        continue;
    }
  }

  funcOp->setAttr(kResultFoldArgAttrName,
                  OpBuilder(funcOp).getI64ArrayAttr(resultArgumentFolding));

  OpBuilder b(returnOp);
  SmallVector<Value> returnValues;
  for (auto en : enumerate(resultArgumentFolding)) {
    LLVM_DEBUG(DBGS() << "return idx: " << en.index() << " folds on "
                      << en.value() << "\n");
    // Return value folds on some input.
    if (en.value() >= 0) continue;

    // Return value does not fold, add it to the new return op.
    Value unfolded = returnOp->getOperand(en.index());
    if (auto tensorLoadOp = unfolded.getDefiningOp<TensorLoadOp>()) {
      unfolded = tensorLoadOp.memref();
      for (Operation *user : llvm::make_early_inc_range(unfolded.getUsers()))
        if (isa<DeallocOp>(user)) user->erase();
    }
    returnValues.push_back(unfolded);
    llvm::errs() << "return val does not fold: " << returnValues.back() << "\n";
  }
  b.create<ReturnOp>(returnOp.getLoc(), returnValues);
  returnOp->erase();

  auto argTypes = llvm::to_vector<4>(
      llvm::map_range(funcOp.getArguments(), [](BlockArgument bbArg) -> Type {
        // TODO: non-zero address space.
        // TODO: layout information if relevant.
        if (auto tensorType = bbArg.getType().dyn_cast<RankedTensorType>())
          return getContiguousMemRefType(tensorType);
        return bbArg.getType();
      }));
  funcOp.setType(FunctionType::get(funcOp->getContext(), argTypes,
                                   ValueRange{returnValues}.getTypes()));
  Block &frontBlock = funcOp.body().front();
  for (unsigned idx = 0, e = frontBlock.getNumArguments(); idx < e; ++idx) {
    auto bbArg = frontBlock.getArgument(0);
    auto tensorType = bbArg.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) {
      frontBlock.addArgument(bbArg.getType());
      bbArg.replaceAllUsesWith(frontBlock.getArguments().back());
    } else {
      // TODO: non-zero address space.
      // TODO: layout information if relevant.
      Value memref =
          frontBlock.addArgument(getContiguousMemRefType(tensorType));
      OpBuilder b(funcOp->getContext());
      // No InsertionGuard needed here.
      b.setInsertionPointToStart(&frontBlock);
      Value tensor = b.create<TensorLoadOp>(funcOp->getLoc(), memref);
      bbArg.replaceAllUsesWith(tensor);
    }
    frontBlock.eraseArgument(0);
  }

  LLVM_DEBUG(DBGS() << "End bufferizeFunctionCallBoundaries:\n" << funcOp);
}

/// Bufferize a single function call.
/// Look for the following pattern for each result to determine whether it can
/// fold onto an argument:
/// ```
///    func @foo(%A: tensor<...>, ..., %Z: tensor<...>) ->
///      (tensor<...>, ..., tensor<...>)
///      #inplace_attr_specification
///    {
///       %p = tensor_to_memref(%some_arg): ...
///       ... // uses of %p (read or writes)
///       %t = tensor_load %p: ...
///       return ..., %t, ...: ..., tensor<...>, ...
///    }
/// ```
static void bufferizeFunctionCall(CallOpInterface callOp,
                                  DominanceInfo &domInfo) {
  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp) return;
  if (funcOp.body().empty()) return;

  // Only bufferizes the first time `funcOp` is encountered.
  bufferizeFunctionCallBoundaries(funcOp);

  SmallVector<Value> newOperands;
  for (Value v : callOp->getOperands()) {
    if (!v.getType().isa<RankedTensorType>()) {
      newOperands.push_back(v);
      continue;
    }
    if (auto tensorLoadOp = v.getDefiningOp<TensorLoadOp>()) {
      newOperands.push_back(tensorLoadOp.memref());
      continue;
    }
    llvm::errs() << "operand: " << v << "\n";
    llvm_unreachable("Operand does not come from a tensor_load");
  }

  assert(isa<CallOp>(callOp.getOperation()) && "expected a CallOp");
  OpBuilder b(callOp);
  Operation *newCallOp = b.create<CallOp>(
      callOp.getLoc(), funcOp.sym_name(),
      funcOp.type().cast<FunctionType>().getResults(), newOperands);
  newCallOp->setAttrs(callOp.getAttrs());

  int numFoldedArgsSoFar = 0;
  for (unsigned callRetIdx = 0, e = callOp->getNumResults(); callRetIdx < e;
       ++callRetIdx) {
    unsigned newCallReturnIdx = callRetIdx - numFoldedArgsSoFar;
    auto maybeFoldedArgIndex = getResultFoldArgIndex(funcOp, callRetIdx);
    if (maybeFoldedArgIndex) ++numFoldedArgsSoFar;

    // If not a ranked tensor, no changes, just replace the new result.
    if (!callOp->getResult(callRetIdx).getType().isa<RankedTensorType>()) {
      assert(!maybeFoldedArgIndex);
      callOp->getResult(callRetIdx)
          .replaceAllUsesWith(newCallOp->getResult(newCallReturnIdx));
      continue;
    }

    // If the old callOp result is a ranked tensor that does not fold on some
    // input, then there must be an allocated return value.
    // That value should be deallocated by the caller.
    // That value should be lifted out of the callee at the first enclosing
    // parallel scope. This lifting should be done to (the meet of) all
    // callers before we can hoist the alloc out of the funcOp.
    Value resultMemref = (maybeFoldedArgIndex)
                             ? newOperands[*maybeFoldedArgIndex]
                             : newCallOp->getResult(newCallReturnIdx);
    callOp->getResult(callRetIdx)
        .replaceAllUsesWith(
            b.create<TensorLoadOp>(callOp.getLoc(), resultMemref));
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(callOp->getBlock()->getTerminator());
    // If function returns a memref, it must be freed.
    if (!maybeFoldedArgIndex)
      b.create<DeallocOp>(callOp.getLoc(), resultMemref);
  }

  callOp->erase();
}

//===----------------------------------------------------------------------===//
// Bufferization passes.
//===----------------------------------------------------------------------===//

// Transformations that run iteratively with bufferization.
void LinalgComprehensiveBufferizePass::runEnablingTransforms(FuncOp funcOp) {
  if (failed(runPipeline(enablingPassPipeline, funcOp)))
    return signalPassFailure();
  (void)runPipeline(enablingPassPipeline, funcOp);
  linalg::hoistRedundantVectorTransfers(funcOp);
  linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
}

void LinalgComprehensiveBufferizePass::bufferizeFuncOpInternals(FuncOp funcOp) {
  LLVM_DEBUG(DBGS() << "Start BufferizeFuncOpInternals:\n" << funcOp);

  OpBuilder b(funcOp->getContext());
  BlockAndValueMapping bvm;
  bool changed = true;
  // It is likely overkill to do this in a loop with canonicalization and
  // hoisting but until we stabilize bufferization, c'est la vie.
  while (changed) {
    changed = false;
    runEnablingTransforms(funcOp);

    // CSE changes the result of the analysis, need to compute/mark/invalidate
    // at each iteration.
    inplaceAnalysisFuncOpInternals(funcOp);
    auto guard = llvm::make_scope_exit([&] {
      funcOp.walk([&](Operation *op) { op->removeAttr(kInPlaceAttrName); });
    });

    funcOp.walk([&](Operation *operation) {
      llvm::TypeSwitch<Operation *, void>(operation)
          // TensorLoadOp is not allowed to just fold into the memref!
          // If it may alias, it must clone.
          .Case([&](TensorLoadOp op) {
            // TODO: reduce amount of surprise.
            if (auto tensorToMemRef =
                    op.memref().getDefiningOp<TensorToMemrefOp>()) {
              // Folding is allowed thwn tensor_to_memref immediately
              // precedes tensor_load -> no interleaved aliasing.
              if (tensorToMemRef->getNextNode() == op) {
                map(bvm, op.result(), op.memref());
                changed = true;
              }
              // TODO: else clone.
            }
          })
          .Case([&](TensorToMemrefOp op) {
            // TODO: reduce amount of surprise.
            Value repl = bvm.lookupOrDefault(op.tensor());
            if (op.memref() != repl) {
              op.memref().replaceAllUsesWith(repl);
              op->erase();
            }
          })
          .Case([&](InitTensorOp op) {
            changed = succeeded(convertInitTensorOp(b, op, bvm));
          })
          .Case([&](SubTensorOp op) {
            changed = succeeded(convertSubTensorOp(b, op, bvm));
          })
          .Case([&](SubTensorInsertOp op) {
            changed = succeeded(convertSubTensorInsertOp(b, op, bvm));
          })
          .Case([&](tensor::CastOp op) {
            changed = succeeded(convertTensorCastOp(b, op, bvm));
          })
          .Case([&](PadTensorOp op) {
            changed = succeeded(convertPadTensorOp(b, op, bvm));
          })
          .Case([&](LinalgOp op) {
            changed = succeeded(convertAnyLinalgOp(b, op, bvm));
          })
          .Case([&](VectorTransferOpInterface op) {
            changed = succeeded(convertTransferOp(b, op, bvm));
          });
    });

    LLVM_DEBUG(DBGS() << "BufferizeFuncOpInternals step:\n" << funcOp);
  }
}

namespace mlir {
std::unique_ptr<Pass> createLinalgComprehensiveBufferizePass() {
  return std::make_unique<LinalgComprehensiveBufferizePass>();
}
namespace linalg {
void registerLinalgComprehensiveBufferizePass() {
  PassRegistration<LinalgComprehensiveBufferizePass> pass(
      "linalg-comprehensive-bufferize-inplace",
      "Perform all required bufferization incantations to convert code with "
      "Linalg ops on tensors to buffers with inplace optimizations.");
}
}  // namespace linalg
}  // namespace mlir

void LinalgComprehensiveBufferizePass::runOnOperation() {
  ModuleOp module = getOperation();
  DominanceInfo domInfo(module);
  module.walk([&](CallOpInterface callOp) {
    inplaceFunctionArgumentAnalysis(callOp, domInfo);
  });

  module.walk([&](FuncOp funcOp) { bufferizeFuncOpInternals(funcOp); });

  // Recompute domInfo.
  domInfo = DominanceInfo(module);
  module.walk(
      [&](CallOpInterface callOp) { bufferizeFunctionCall(callOp, domInfo); });
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  (void)pm.run(module);

  // Cleanups and sanity checks.
  module.walk([&](Operation *op) {
    op->removeAttr(kInPlaceAttrName);
    op->removeAttr(kResultFoldArgAttrName);
    if (auto tensorLoadOp = dyn_cast<TensorLoadOp>(op)) {
      if (tensorLoadOp.memref().getDefiningOp<TensorToMemrefOp>()) {
        op->getParentOfType<ModuleOp>()->dump();
        op->emitWarning(
            "Most likely incorrect pattern: tensor_load(tensor_to_memref)");
        abort();
      }
    }
    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      for (auto result : callOp->getResults()) {
        if (result.getType().isa<MemRefType>()) {
          op->getParentOfType<ModuleOp>()->dump();
          op->emitWarning(
              "Most likely incorrect pattern: function returning memref -> "
              "alloc needs to be hoisted out of function boundary");
          abort();
        }
      }
    }
  });
}
