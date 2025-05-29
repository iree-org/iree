#include "iree/compiler/AssertInserter/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_ASSERTINBOUNDSPASS
#include "iree/compiler/AssertInserter/Passes.h.inc"

// TODO: this does not belong here!
#define GEN_PASS_DEF_CHECKSTATICASSERTIONSPASS
#include "iree/compiler/AssertInserter/Passes.h.inc"
} // namespace mlir::iree_compiler

using namespace mlir;

/// Kind of checks to insert.
enum class CheckKind { PerDimension, Combined };

// These should become an interface eventually, but currently not worth the
// complexity of adding one.

/// Whether the operation is supposed by the assertion inserter.
static bool isSupported(memref::LoadOp) { return true; }
static bool isSupported(memref::StoreOp) { return true; }
static bool isSupported(vector::LoadOp loadOp) {
  return !loadOp.getVectorType().isScalable();
}
static bool isSupported(vector::StoreOp storeOp) {
  return !storeOp.getVectorType().isScalable();
}

/// Returns the number of elements accessed along the given dimension by a
/// vector load/store operation.
template <typename OpTy>
static int64_t getAccessExtentAlongDimVectorOp(OpTy op, unsigned dim) {
  static_assert(llvm::is_one_of<OpTy, vector::LoadOp, vector::StoreOp>::value,
                "expected vector load or store");
  if (isa<VectorType>(op.getMemRefType().getElementType()))
    return 1;
  unsigned leadingOneDims =
      op.getMemRefType().getRank() - op.getVectorType().getRank();
  return dim < leadingOneDims
             ? 1
             : op.getVectorType().getDimSize(dim - leadingOneDims);
}

/// Returns the number of elements accessed along the given dimension by the
/// operation.
static int64_t getAccessExtentAlongDim(memref::LoadOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(memref::StoreOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(vector::LoadOp loadOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(loadOp, dim);
}
static int64_t getAccessExtentAlongDim(vector::StoreOp storeOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(storeOp, dim);
}

/// Returns the base memref that is being indexed into by the accessing
/// operation.
static Value getAccessBase(memref::LoadOp loadOp) { return loadOp.getMemRef(); }
static Value getAccessBase(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}
static Value getAccessBase(vector::LoadOp loadOp) { return loadOp.getBase(); }
static Value getAccessBase(vector::StoreOp storeOp) {
  return storeOp.getBase();
}

// End pseudo-interface.

/// Inserts `cf.assert` checking whether the subscripts of the given
/// memory-accessing operation are in bounds.
template <typename OpTy>
static LogicalResult insertInBoundsAssertions(OpBuilder &builder, OpTy op,
                                              CheckKind checkKind) {
  if (!isSupported(op))
    return op.emitError() << "unsupported variation of the op";

  ImplicitLocOpBuilder b(op->getLoc(), builder);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(0);
  Value totalCheck =
      checkKind == CheckKind::Combined
          ? b.createOrFold<arith::ConstantIntOp>(1, b.getI1Type())
          : nullptr;
  for (unsigned i = 0, e = op.getMemRefType().getRank(); i < e; ++i) {
    Value index = b.createOrFold<arith::ConstantIndexOp>(i);
    Value dim = b.createOrFold<memref::DimOp>(getAccessBase(op), index);
    Value subscript = op.getIndices()[i];
    Value lowerBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::sge, subscript, zero);

    int64_t accessExtent = getAccessExtentAlongDim(op, i);
    assert(accessExtent >= 1 && "expected positive access extent");
    Value lastAccessedIndex =
        accessExtent == 1
            ? subscript
            : b.createOrFold<arith::AddIOp>(
                  subscript,
                  b.createOrFold<arith::ConstantIndexOp>(accessExtent - 1));

    Value upperBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::slt, lastAccessedIndex, dim);
    Value boundCheck =
        b.createOrFold<arith::AndIOp>(lowerBoundCheck, upperBoundCheck);
    if (checkKind == CheckKind::PerDimension) {
      b.createOrFold<cf::AssertOp>(
          boundCheck,
          "memref access out of bounds along dimension " + std::to_string(i));
    } else {
      assert(checkKind == CheckKind::Combined && "unsupported check kind");
      totalCheck = b.createOrFold<arith::AndIOp>(totalCheck, boundCheck);
    }
  }
  if (checkKind == CheckKind::Combined) {
    b.createOrFold<cf::AssertOp>(totalCheck, "memref access out of bounds");
  }
  return success();
}

namespace {
/// Options for the in-bound assertion inserter.
struct InsertInBoundsAssertionsConfig {
  CheckKind checkKind;
  bool includeVectorLoadStore;
  bool warnOnUnknown;
};
} // namespace

/// Inserts in-bounds check for the given operation immediately prior to it.
/// Specific checks depend on the kind of operation and the configuration. Emits
/// errors when checks cannot be inserted and warnings, if requested in the
/// configuration, on unhandled operations that may need checks.
static LogicalResult
insertInBoundsAssertionDispatch(OpBuilder &builder, Operation *op,
                                const InsertInBoundsAssertionsConfig &config) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<memref::LoadOp, memref::StoreOp>([&](auto casted) {
        return insertInBoundsAssertions(builder, casted, config.checkKind);
      })
      .Case<vector::LoadOp, vector::StoreOp>([&](auto casted) {
        // Vector load/store specifically allow for lowering-defined
        // out-of-bounds access when using scalar-typed memory. Ignore
        // those unless explicitly requested by the caller.
        if (!config.includeVectorLoadStore &&
            !isa<VectorType>(casted.getMemRefType().getElementType()))
          return success();

        return insertInBoundsAssertions(builder, casted, config.checkKind);
      })
      .Default([&](Operation *uncasted) {
        if (!config.warnOnUnknown)
          return success();

        auto effecting = dyn_cast<MemoryEffectOpInterface>(uncasted);
        if (!effecting)
          return success();

        SmallVector<MemoryEffects::EffectInstance> effects;
        effecting.getEffects(effects);
        if (llvm::none_of(effects, [](MemoryEffects::EffectInstance &instance) {
              bool effectMayBeOnMemRef =
                  !instance.getValue() ||
                  isa<MemRefType>(instance.getValue().getType());
              return effectMayBeOnMemRef &&
                     isa<MemoryEffects::Read, MemoryEffects::Write>(
                         instance.getEffect());
            }))
          return success();

        uncasted->emitWarning()
            << "operation with memory effects was not processed";
        return success();
      });
}

/// Applies `func` to every element in `range` and checks the result. Fails
/// immediately if application failed on one element without checking the
/// following elements.
template <typename Range, typename Func>
LogicalResult try_transform(Range &&range, Func &&func) {
  for (auto &&elem : range) {
    if (failed(func(elem)))
      return failure();
  }
  return success();
}

// TODO: these should be part of the function op interface. Note the caveat for
// the dependent dialects of the pass using it if the call does not belong to
// the same dialect as the func op (very unlikely).

/// Creates a call to the given function with the provided operands. The
/// operands are expected to be compatible with the function. The builder must
/// have an appropriate insertion point set up.
static Operation *createCall(OpBuilder &builder, Location loc,
                             FunctionOpInterface func, ValueRange operands) {
  if (auto funcFunc = dyn_cast<func::FuncOp>(*func)) {
    return builder.create<func::CallOp>(loc, funcFunc, operands);
  }
  func.emitError() << "cannot create a call to this function";
  return nullptr;
}

/// Creates a return from the given function with the provided operands. The
/// operands are expected to be compatible with the function results. The
/// builder must have an appropriate insertion point set up.
static Operation *createReturn(OpBuilder &builder, Location loc,
                               FunctionOpInterface func, ValueRange operands) {
  if (auto funcFunc = dyn_cast<func::FuncOp>(*func)) {
    return builder.create<func::ReturnOp>(loc, operands);
  }
  func.emitError() << "cannot create a return from this function";
  return nullptr;
}

static LogicalResult isSpeculatableForInBoundsAssertions(Operation *op);

/// Checks if all operations in the provided ranges can be cloned to the
/// speculative assertion function.
template <typename Range>
LogicalResult checkSpeculatable(Range &&range) {
  for (Operation *element : range) {
    if (failed(isSpeculatableForInBoundsAssertions(element)))
      return failure();
  }
  return success();
}

/// Checks if the given operation can be cloned to the speculative assertion
/// function. Unlike the generic speculatability check, this accounts for the
/// fact that some operations will not be cloned.
static LogicalResult isSpeculatableForInBoundsAssertions(Operation *op) {
  // We consider structured control flow speculatable if the values it yields
  // come from speculatable operations. Note that this also means the backward
  // slice of the yielded values must be included into the operations to clone.
  // This is the case because we include the slice of the terminator.
  //
  // TODO: We may want to have a mechanism to clone control flow without
  // yielding additional values. However, we may need to still yield the values
  // if they are used by some operations we are carrying over to the speculative
  // function. So potentially the current solution is to keep them and to later
  // run a dead iterarg elimination procedure followed by regular DCE in the
  // body.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (!forOp.getBody()->mightHaveTerminator())
      return success();
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.filter = [&](Operation *filterOp) { return op != filterOp; };
    getBackwardSlice(forOp.getBody()->getTerminator(), &slice);
    slice.remove(op);
    return checkSpeculatable(slice);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp.getElseRegion().empty() ||
        !ifOp.getThenRegion().front().mightHaveTerminator() ||
        !ifOp.getElseRegion().front().mightHaveTerminator())
      return success();

    SetVector<Operation *> slice, sliceElse;
    BackwardSliceOptions options;
    options.filter = [&](Operation *filterOp) { return op != filterOp; };
    getBackwardSlice(ifOp.getThenRegion().front().getTerminator(), &slice,
                     options);
    slice.remove(op);
    if (failed(checkSpeculatable(slice)))
      return failure();
    getBackwardSlice(ifOp.getElseRegion().front().getTerminator(), &sliceElse,
                     options);
    sliceElse.remove(op);
    return checkSpeculatable(sliceElse);
  }

  // For other operations, defer to the generic speculability check.
  return success(isSpeculatable(op));
}

/// Creates a new function performing a speculative check that all known kinds
/// of memory accesses in the given function are in bounds, and inserts a call
/// to this function as the first operation in the given function. Reports
/// errors and returns failure if such a check cannot be created, in particular
/// if it would require speculating operations that should not be, such as
/// stores to memory.
static LogicalResult
createSpeculativeInBoundsChecks(RewriterBase &rewriter,
                                FunctionOpInterface funcOp,
                                const InsertInBoundsAssertionsConfig &config) {
  SetVector<Operation *> operationsToProcess;
  llvm::SmallPtrSet<Operation *, 8> checkedOperations;

  // Collects the backward slice of the given `source` (operation or value) into
  // `operationsToProcess`. Checks that the operations in the slice can be
  // speculated for the purposes of in-bounds checks insertion.
  auto collectSlice = [&](auto source, bool inclusive) {
    if constexpr (std::is_same_v<std::remove_reference_t<decltype(source)>,
                                 Value>) {
      assert(!inclusive && "backward slice of a value cannot be inclusive");
    }
    SetVector<Operation *> slice;
    BackwardSliceOptions sliceOptions;
    sliceOptions.inclusive = inclusive;
    getBackwardSlice(source, &slice, sliceOptions);
    size_t originalSize = operationsToProcess.size();
    operationsToProcess.insert_range(slice);

    // Check that newly inserted ops are speculatable.
    for (Operation *op :
         operationsToProcess.getArrayRef().drop_front(originalSize)) {
      if (failed(isSpeculatableForInBoundsAssertions(op))) {
        op->emitError() << "in-bounds check generation requires speculating "
                           "this operation, but it is not speculatable";
        return failure();
      }
    }

    return success();
  };
  auto collectSlices = [&](ValueRange values) {
    return try_transform(
        values, [&](Value v) { return collectSlice(v, /*inclusive=*/false); });
  };

  // Returns `true` if the given operation is a terminator returning from the
  // function being processed.
  auto isFuncReturn = [&](Operation *op) {
    return op->hasTrait<OpTrait::IsTerminator>() &&
           op->getNumSuccessors() == 0 && op->getParentOp() == funcOp;
  };

  // Collect operations to be processed: either cloned to the speculative check
  // function or replaced with assertions.
  WalkResult walkResult = funcOp->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      if (region.empty() || llvm::hasSingleElement(region.getBlocks()))
        continue;

      // TODO: support branching control flow, which requires adding support for
      // it in backward slicing.
      op->emitError("branching control flow not supported");
      return WalkResult::interrupt();
    }
    auto result =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<memref::LoadOp, vector::LoadOp>([&](Operation *op) {
              // The slice includes the address of the load to account for
              // situations where (a) it comes from a view/subview and we want
              // to replicate that and (b) avoid replicating allocations/copies
              // thanks to the speculability check.
              if (failed(collectSlice(op, /*inclusive=*/false)))
                return failure();
              // The current operation is not included in the backward slice so
              // add it separately. This is intentional because the
              // memory-accessing operation is known to be non-speculatable
              // while operations we collect are checked to be speculatable. We
              // will not be actually cloning it, so don't check it.
              operationsToProcess.insert(op);
              checkedOperations.insert(op);
              return success();
            })
            .Case<memref::StoreOp, vector::StoreOp>([&](auto op) {
              // Do not include the value to store in the slice, we don't need
              // it since we are not going to actually store it.
              if (failed(collectSlices(op.getIndices())))
                return failure();
              if (failed(collectSlices(getAccessBase(op))))
                return failure();
              // Same as above for loads.
              operationsToProcess.insert(op);
              checkedOperations.insert(op);
              return success();
            })
            .Case<scf::IfOp, scf::ForOp>([&](Operation *op) {
              if (failed(collectSlice(op, /*inclusive=*/true)))
                return failure();
              return success();
            })
            .Case<FunctionOpInterface>([](auto op) { return success(); })
            .Default([&](Operation *op) {
              if (op->getNumRegions() != 0) {
                // TODO: we could default to computing backwards slices of all
                // operands conservatively, but there may also be side effects
                // affecting the control flow...
                op->emitError() << "unhandled region-carrying operation";
                return failure();
              }
              if (!op->hasTrait<OpTrait::IsTerminator>())
                return success();

              // If we have a terminator "returning" from the current function,
              // i.e. has no successors, it will be cloned without operands, so
              // don't take its backward slice, but include the op itself for
              // cloning.
              if (isFuncReturn(op)) {
                operationsToProcess.insert(op);
                return success();
              }

              return collectSlice(op, /*inclusive=*/true);
            });

    if (failed(result))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // Create the speculative check function.
  IRMapping mapping;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(funcOp);
  auto clonedFuncOp =
      cast<FunctionOpInterface>(rewriter.cloneWithoutRegions(*funcOp, mapping));
  clonedFuncOp.setName("__speculative_in_bounds_check_" +
                       clonedFuncOp.getName().str());
  clonedFuncOp.setVisibility(SymbolTable::Visibility::Private);
  if (failed(clonedFuncOp.eraseResults(
          llvm::BitVector(funcOp.getNumResults(), true)))) {
    return funcOp->emitError()
           << "cannot create a speculative checker function with no results to "
              "match this function";
  }

  // Clone operations into the speculative function and insert checks.
  Block *entryBlock = clonedFuncOp.addEntryBlock();
  mapping.map(&funcOp.getFunctionBody().front(), entryBlock);
  mapping.map(funcOp.getArguments(), entryBlock->getArguments());
  // Need to re-sort operations because they come from distinct slices and may
  // be unordered wrt each other even if they were ordered in one slice.
  // Operations that are being checked will not be actually cloned, but we
  // still want them in the topological order so we can insert the checks at
  // the appropriate location.
  for (Operation *op : topologicalSort(operationsToProcess)) {
    // Given that ops appear in topological order, we can always insert them to
    // the end of the corresponding block.
    rewriter.setInsertionPointToEnd(mapping.lookup(op->getBlock()));

    // Insert the checks where required. Note that insertion operates around an
    // existing memory-accessing operation, so we first clone it, call the
    // insertion, and then erase the cloned operation.
    if (checkedOperations.contains(op)) {
      Operation *cloned = rewriter.clone(*op, mapping);
      if (failed(insertInBoundsAssertionDispatch(rewriter, cloned, config)))
        return failure();
      rewriter.eraseOp(cloned);
      continue;
    }

    // Create explicit no-operand returns from the function since we removed all
    // results.
    if (isFuncReturn(op)) {
      if (createReturn(rewriter, op->getLoc(), funcOp, /*operands=*/{}) ==
          nullptr)
        return failure();
      continue;
    }

    // Otherwise clone the operation. Create blocks for regions when
    // appropriate, but don't clone the regions themselves. They will be
    // populated later.
    Operation *cloned = rewriter.cloneWithoutRegions(*op, mapping);
    for (auto &&[origRegion, cloneRegion] :
         llvm::zip_equal(op->getRegions(), cloned->getRegions())) {
      if (origRegion.empty())
        continue;

      assert(llvm::hasSingleElement(origRegion) &&
             "hitting this means the support for branching control flow is "
             "incomplete");
      Block &entryBlock = origRegion.front();
      SmallVector<Location> locations =
          llvm::map_to_vector(entryBlock.getArguments(),
                              [](Value value) { return value.getLoc(); });
      Block *clonedEntryBlock =
          rewriter.createBlock(&cloneRegion, /*insertPt=*/{},
                               entryBlock.getArgumentTypes(), locations);
      mapping.map(&entryBlock, clonedEntryBlock);
      mapping.map(entryBlock.getArguments(), clonedEntryBlock->getArguments());
    }
  }

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  Operation *call = createCall(rewriter, clonedFuncOp->getLoc(), clonedFuncOp,
                               funcOp.getArguments());
  return success(call != nullptr);
}

/// Inserts bounds check assertions in place.
static LogicalResult
insertInPlaceInBoundsAssertions(Operation *root,
                                const InsertInBoundsAssertionsConfig &config) {
  OpBuilder builder(root->getContext());
  WalkResult walkResult = root->walk([&](Operation *op) {
    OpBuilder::InsertionGuard raii(builder);
    builder.setInsertionPoint(op);

    if (failed(insertInBoundsAssertionDispatch(builder, op, config)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

/// For all functions in the given scope, creates new functions speculatively
/// checking bounds for all known memory-accessing operations, and inserts calls
/// to those functions.
static LogicalResult insertSpeculativeInBoundsAssertions(
    Operation *root, const InsertInBoundsAssertionsConfig &config) {
  IRRewriter rewriter(root->getContext());
  WalkResult walkResult =
      root->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
        if (failed(createSpeculativeInBoundsChecks(rewriter, funcOp, config)))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
  return failure(walkResult.wasInterrupted());
}

namespace {
class AssertInBoundsPass
    : public mlir::iree_compiler::impl::AssertInBoundsPassBase<AssertInBoundsPass> {
public:
  using mlir::iree_compiler::impl::AssertInBoundsPassBase<AssertInBoundsPass>::AssertInBoundsPassBase;

  void runOnOperation() override;
};

void AssertInBoundsPass::runOnOperation() {
  InsertInBoundsAssertionsConfig config;
  config.checkKind =
      checkEachDim ? CheckKind::PerDimension : CheckKind::Combined;
  config.includeVectorLoadStore = includeVectorLoadStore;
  config.warnOnUnknown = warnOnUnknown;

  if (!createSpeculativeFuncs) {
    if (failed(insertInPlaceInBoundsAssertions(getOperation(), config)))
      signalPassFailure();
    return;
  }

  if (failed(insertSpeculativeInBoundsAssertions(getOperation(), config)))
    signalPassFailure();
}

class CheckStaticAssertionsPass
    : public mlir::iree_compiler::impl::CheckStaticAssertionsPassBase<
          CheckStaticAssertionsPass> {
public:
  void runOnOperation() override {
    WalkResult walkResult = getOperation()->walk([&](cf::AssertOp assertOp) {
      APInt value;
      if (matchPattern(assertOp.getArg(), m_ConstantInt(&value)) &&
          value.isZero()) {
        assertOp->emitError() << "assertion known to be false";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
