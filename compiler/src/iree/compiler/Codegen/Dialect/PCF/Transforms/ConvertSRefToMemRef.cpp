// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/ConversionDialectInterface.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-pcf-convert-sref-to-memref"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_CONVERTSREFTOMEMREFPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"
namespace {

class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<PCFConversionDialectInterface>(dialect);
      if (!iface) {
        continue;
      }
      iface->loadSRefLoweringDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

struct ConvertSRefToMemRefPass final
    : impl::ConvertSRefToMemRefPassBase<ConvertSRefToMemRefPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // Direct dialect deps.
    registry.insert<iree_compiler::IREE::Codegen::IREECodegenDialect,
                    iree_compiler::IREE::PCF::PCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Layout Propagation Analysis Impl
//===----------------------------------------------------------------------===//

static bool isDefaultOrStrided(Attribute layout) {
  auto mapAttr = dyn_cast<AffineMapAttr>(layout);
  return !layout || (mapAttr && mapAttr.isIdentity()) ||
         isa<StridedLayoutAttr>(layout);
}

// State that tracks memref layouts per value. Lattice description is below
// next to the state fields.
struct StridedLayoutState : DFX::AbstractState {
  bool isValidState() const override { return isValid; }
  bool isAtFixpoint() const override { return !isValid || isFinalized; }

  void invalidate() { isValid = false; }

  ChangeStatus indicateOptimisticFixpoint() override {
    isFinalized = true;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    isFinalized = true;
    assumed = MemRefType();
    return ChangeStatus::CHANGED;
  }

  MemRefType getAssumed() const { return assumed; }

  // Resets the assumed value to the given value. This does no unioning and
  // assumes it is a proper fixpoint minimum.
  void setAssumed(MemRefType newAssumed) { assumed = newAssumed; }

  // "Clamps" this state with |rhs|. The assumed value will contain the matching
  // static strides of both assumed layouts, or dynamic if sizes don't match.
  void operator^=(const StridedLayoutState &rhs) {
    // Ignore if this state is already at a fixed point. In this case it should
    // be a pessimistic fixed point as all optimistic fixed points are
    // determined on initialization/first update. Also nothing to do on an
    // invalid state.
    if (isFinalized || !isValid) {
      return;
    }

    if (!rhs.isValidState()) {
      return invalidate();
    }

    // If no value is assumed yet, take RHS.
    if (!assumed) {
      assumed = rhs.getAssumed();
      return;
    }

    if (!rhs.getAssumed()) {
      // Two possibilities if the rhs is undefined. First, rhs is at a
      // pessimistic fixed point, in which case we take it.
      if (rhs.isAtFixpoint()) {
        assumed = rhs.getAssumed();
      }
      // Otherwise nothing to do.
      return;
    }

    // Invalidate if memory space mismatches. We could allow for falling back to
    // a more generic memory space but this is in all cases today going to arise
    // from an earlier uncaught failure.
    if (rhs.getAssumed().getMemorySpace() != assumed.getMemorySpace()) {
      return invalidate();
    }

    // Shape and element type should be guaranteed because the sref type
    // carries them so we assert instead.
    assert(rhs.getAssumed().getShape() == assumed.getShape() &&
           rhs.getAssumed().getElementType() == assumed.getElementType() &&
           "Unexpected shape or element type mismatch");

    MemRefLayoutAttrInterface newLayout = MemRefLayoutAttrInterface();
    if (!rhs.getAssumed().getLayout()) {
      if (!isDefaultOrStrided(assumed.getLayout())) {
        // Fail if there is a non-strided and non-default layout.
        return invalidate();
      }
      newLayout = assumed.getLayout();
    } else if (!assumed.getLayout()) {
      if (!isDefaultOrStrided(rhs.getAssumed().getLayout())) {
        // Same here.
        return invalidate();
      }
      newLayout = assumed.getLayout();
    } else {
      // Union the strided layouts.
      auto rhsStridedLayout =
          dyn_cast_if_present<StridedLayoutAttr>(rhs.getAssumed().getLayout());
      auto thisStridedLayout =
          dyn_cast_if_present<StridedLayoutAttr>(assumed.getLayout());
      if (!rhsStridedLayout || !thisStridedLayout ||
          rhsStridedLayout.getStrides().size() !=
              thisStridedLayout.getStrides().size()) {
        return invalidate();
      }

      auto dynamicizeIfUnequal = [](int64_t l, int64_t r) {
        return l != r ? ShapedType::kDynamic : l;
      };

      SmallVector<int64_t> newStrides(thisStridedLayout.getStrides());
      int64_t newOffset = dynamicizeIfUnequal(thisStridedLayout.getOffset(),
                                              rhsStridedLayout.getOffset());

      for (auto [lStride, rStride] :
           llvm::zip_equal(newStrides, rhsStridedLayout.getStrides())) {
        lStride = dynamicizeIfUnequal(lStride, rStride);
      }

      newLayout = StridedLayoutAttr::get(thisStridedLayout.getContext(),
                                         newOffset, newStrides);
    }
    assumed = MemRefType::get(assumed.getShape(), assumed.getElementType(),
                              newLayout, assumed.getMemorySpace());
  }

private:
  // Assumed state for the value. Since shape consistency is already carried on
  // the !pcf.sref type, only the layout and memory space is relevant for the
  // analysis. The state starts uninitialized and is progressively unioned with
  // all producers under the following cases:
  // 1. If this state is uninitialized, take the incoming state.
  // 2. If the incoming state is uninitialized, do nothing.
  // 3. Else update the state with the minimally dynamic layout covering both
  //    this and the incoming state.
  // The lattice first initializes all values and then moves towards the least
  // dynamic fixed point that satisfies all incoming memref types.
  MemRefType assumed = MemRefType();
  // Indicates whether the state is at a fixed point. Only set for values that
  // have a locally determined type (e.g. allocs and tied args).
  bool isFinalized = false;
  // Indicates whether the state is valid. Invalid states are irrecoverable
  // and result in a conversion failure. This happens on:
  // 1. Irreconcilable memref layouts (e.g. different ranks).
  // 2. Non-strided layouts.
  // 3. Conflicting memory spaces.
  // 4. Failure to converge to a fixed point.
  bool isValid = true;
};

// Attribute known floating point range and flags to an IR Value.
class StridedLayoutValueElement
    : public DFX::StateWrapper<StridedLayoutState, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<StridedLayoutState, DFX::ValueElement>;
  using BaseType::BaseType;

  static StridedLayoutValueElement &createForPosition(const Position &pos,
                                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) StridedLayoutValueElement(pos));
  }

  // Identity definitions.
  static const char ID;
  const std::string getName() const override {
    return "StridedLayoutValueElement";
  }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  const std::string getAsStr(AsmState &asmState) const override;

private:
  void initializeValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override;
  ChangeStatus updateOpResult(OpResult result, StridedLayoutState &newState,
                              DFX::Solver &solver);
  ChangeStatus updateBlockArgument(BlockArgument value,
                                   StridedLayoutState &newState,
                                   DFX::Solver &solver);
};
const char StridedLayoutValueElement::ID = 0;

void StridedLayoutValueElement::initializeValue(Value value,
                                                DFX::Solver &solver) {
  if (!isa<PCF::ShapedRefType>(value.getType())) {
    indicatePessimisticFixpoint();
    return;
  }
}

ChangeStatus StridedLayoutValueElement::updateOpResult(
    OpResult result, StridedLayoutState &newState, DFX::Solver &solver) {
  llvm::TypeSwitch<Operation *, void>(result.getOwner())
      .Case([&](PCF::AllocOp allocOp) {
        PCF::ShapedRefType resultType = allocOp.getResultType();
        FailureOr<Attribute> memSpace =
            resultType.getScope().getAllocMemSpace(allocOp.getContext());
        if (failed(memSpace)) {
          allocOp->emitOpError("failed to get memory space for allocation");
          newState.invalidate();
          return;
        }
        newState.setAssumed(
            MemRefType::get(resultType.getShape(), resultType.getElementType(),
                            MemRefLayoutAttrInterface{}, memSpace.value()));
        newState.indicateOptimisticFixpoint();
      })
      .Case([&](RegionBranchOpInterface regionOp) {
        // For region branch ops get the result layout from the union of
        // return sites.
        if (solver.getExplorer().walkReturnOperands(
                regionOp.getOperation(), [&](OperandRange returnOperands) {
                  auto returnOperand = returnOperands[result.getResultNumber()];
                  auto returnState =
                      solver.getElementFor<StridedLayoutValueElement>(
                          *this, Position::forValue(returnOperand),
                          DFX::Resolution::REQUIRED);
                  newState ^= returnState;
                  return WalkResult::advance();
                }) == TraversalResult::INCOMPLETE) {
          newState.indicatePessimisticFixpoint();
          return;
        }
      })
      .Case([&](CallOpInterface callOp) {
        // Give up pessimistically on indirect calls.
        if (isa<Value>(callOp.getCallableForCallee())) {
          newState.indicatePessimisticFixpoint();
          return;
        }
        auto targetSymbol = cast<SymbolRefAttr>(callOp.getCallableForCallee());
        auto callableOp = solver.getExplorer()
                              .getSymbolTables()
                              .lookupNearestSymbolFrom<CallableOpInterface>(
                                  callOp, targetSymbol);
        assert(callableOp && "call target not found");
        // For region branch ops get the result layout from the union of
        // return sites.
        if (solver.getExplorer().walkReturnOperands(
                callableOp, [&](OperandRange returnOperands) {
                  auto returnOperand = returnOperands[result.getResultNumber()];
                  auto returnState =
                      solver.getElementFor<StridedLayoutValueElement>(
                          *this, Position::forValue(returnOperand),
                          DFX::Resolution::REQUIRED);
                  newState ^= returnState;
                  return WalkResult::advance();
                }) == TraversalResult::INCOMPLETE) {
          newState.indicatePessimisticFixpoint();
          return;
        }
      })
      .Case([&](Util::OptimizationBarrierOp barrierOp) {
        auto returnState = solver.getElementFor<StridedLayoutValueElement>(
            *this,
            Position::forValue(barrierOp.getOperand(result.getResultNumber())),
            DFX::Resolution::REQUIRED);
        newState ^= returnState;
      });
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

ChangeStatus StridedLayoutValueElement::updateBlockArgument(
    BlockArgument bbArg, StridedLayoutState &newState, DFX::Solver &solver) {
  bool didUpdate = false;
  if (bbArg.getParentBlock()->isEntryBlock()) {
    didUpdate =
        llvm::TypeSwitch<Operation *, bool>(bbArg.getOwner()->getParentOp())
            .Case([&](PCF::GenericOp genericOp) {
              if (genericOp.isRegionRefArg(bbArg)) {
                auto resultType = dyn_cast<MemRefType>(
                    genericOp.getTiedResult(bbArg).getType());
                if (!resultType ||
                    !isDefaultOrStrided(resultType.getLayout())) {
                  genericOp->emitOpError(
                      "unexpected non-strided or default memref result type ")
                      << resultType;
                  newState.invalidate();
                } else {
                  newState.setAssumed(resultType);
                  newState.indicateOptimisticFixpoint();
                }
              } else {
                // pcf.sref arguments must either be result tied or
                // initialized per the verifier.
                assert(genericOp.isInitializedArg(bbArg) &&
                       "unexpected non-initialized arg");
                auto yield = cast<PCF::YieldOp>(
                    genericOp.getInitializer().front().getTerminator());
                auto initializerState =
                    solver.getElementFor<StridedLayoutValueElement>(
                        *this,
                        Position::forValue(
                            yield->getOperand(bbArg.getArgNumber())),
                        DFX::Resolution::REQUIRED);
                newState ^= initializerState;
              }
              return true;
            })
            .Case([&](PCF::LoopOp loopOp) {
              auto resultType =
                  dyn_cast<MemRefType>(loopOp.getTiedResult(bbArg).getType());
              if (!resultType || !isDefaultOrStrided(resultType.getLayout())) {
                loopOp->emitOpError(
                    "unexpected non-strided or default memref result type ")
                    << resultType;
                newState.invalidate();
              } else {
                newState.setAssumed(resultType);
                newState.indicateOptimisticFixpoint();
              }
              return true;
            })
            .Default([&](Operation *) { return false; });
  }
  if (!didUpdate) {
    solver.getExplorer().walkIncomingBranchOperands(
        bbArg.getOwner(),
        [&](Block *sourceBlock, OperandRange operands, size_t offset) {
          auto bbArgState = solver.getElementFor<StridedLayoutValueElement>(
              *this,
              Position::forValue(operands[bbArg.getArgNumber() + offset]),
              DFX::Resolution::REQUIRED);
          newState ^= bbArgState;
          return WalkResult::advance();
        });
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

ChangeStatus StridedLayoutValueElement::updateValue(Value value,
                                                    DFX::Solver &solver) {
  StridedLayoutState newState = getState();

  // Handle op results.
  if (auto result = dyn_cast<OpResult>(value)) {
    return updateOpResult(result, newState, solver);
  }

  // Else it is a block argument.
  return updateBlockArgument(cast<BlockArgument>(value), newState, solver);
}

const std::string
StridedLayoutValueElement::getAsStr(AsmState &asmState) const {
  MemRefType range = getAssumed();
  std::string s("layout: ");
  llvm::raw_string_ostream os(s);
  range.print(os, asmState);
  return s;
}

class SRefLayoutAnalysis {
public:
  explicit SRefLayoutAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::RECURSE),
        solver(explorer, allocator) {
    explorer.initialize();
  }

  AsmState &getAsmState() { return solver.getAsmState(); }
  Explorer &getExplorer() { return explorer; }

  LogicalResult run() {
    // Initialize all shaped ref values to maximally dynamic layouts.
    explorer.walkValuesOfType<PCF::ShapedRefType>([&](Value v) {
      solver.getOrCreateElementFor<StridedLayoutValueElement>(
          Position::forValue(v));
      return WalkResult::advance();
    });

    // Run solver to completion.
    LogicalResult result = solver.run();
    LLVM_DEBUG(solver.print(llvm::dbgs()));
    return result;
  }

  // Returns the memref type this value should be converted to.
  FailureOr<MemRefType> getConvertedType(Value value) {
    assert(isa<PCF::ShapedRefType>(value.getType()) &&
           "unexpected non sref type");
    // memref -> sref conversions come in as unrealized casts.
    if (auto unrealizedCast =
            value.getDefiningOp<UnrealizedConversionCastOp>()) {
      return cast<MemRefType>(unrealizedCast->getOperandTypes()[0]);
    }

    // sref -> memref conversions for op results and produced block args are
    // queried from the analysis.
    auto &stridedLayout =
        solver.getOrCreateElementFor<StridedLayoutValueElement>(
            Position::forValue(value));
    if (!stridedLayout.isValidState() || !stridedLayout.getAssumed()) {
      return failure();
    }
    StridedLayoutState state = stridedLayout.getState();
    return state.getAssumed();
  }

  // Returns the function type this callable should be converted to.
  FailureOr<FunctionType>
  lookupConvertedFunctionArgs(CallableOpInterface callableOp) {
    // Check if we have a cached conversion.
    FunctionType cachedType =
        cachedFunctionTypeMap.lookup_or(callableOp, FunctionType());
    if (cachedType) {
      return cachedType;
    }

    // Since the function op conversion pattern calls this function we're
    // guaranteed the argument/result types will be unconverted when this is
    // first called irrespective of pattern application order.
    SmallVector<Type> argumentTypes;
    Region *region = callableOp.getCallableRegion();
    if (!region) {
      return failure();
    }
    for (BlockArgument bbArg : callableOp.getCallableRegion()->getArguments()) {
      if (!isa<PCF::ShapedRefType>(bbArg.getType())) {
        argumentTypes.push_back(bbArg.getType());
        continue;
      }
      FailureOr<MemRefType> maybeConvertedType = getConvertedType(bbArg);
      if (failed(maybeConvertedType)) {
        return failure();
      }
      argumentTypes.push_back(maybeConvertedType.value());
    }

    int64_t numResultsToConvert = llvm::count_if(
        callableOp.getResultTypes(), llvm::IsaPred<PCF::ShapedRefType>);
    if (numResultsToConvert == 0) {
      return FunctionType::get(callableOp->getContext(), argumentTypes,
                               callableOp.getResultTypes());
    }

    SmallVector<StridedLayoutState> states(numResultsToConvert,
                                           StridedLayoutState());
    if (solver.getExplorer().walkReturnOperands(
            callableOp, [&](OperandRange range) {
              int64_t currState = 0;
              for (Value v : range) {
                if (isa<PCF::ShapedRefType>(v.getType())) {
                  FailureOr<MemRefType> maybeConvertedType =
                      getConvertedType(v);
                  if (failed(maybeConvertedType)) {
                    return WalkResult::interrupt();
                  }
                  StridedLayoutState newState;
                  newState.setAssumed(maybeConvertedType.value());
                  newState.indicateOptimisticFixpoint();
                  states[currState] ^= newState;
                  ++currState;
                }
              }
              return WalkResult::advance();
            }) == TraversalResult::INCOMPLETE) {
      return failure();
    }

    SmallVector<Type> newResultTypes(callableOp.getResultTypes());
    int64_t currState = 0;
    for (auto &type : newResultTypes) {
      if (isa<PCF::ShapedRefType>(type)) {
        auto &state = states[currState];
        if (!state.isValidState() || !state.getAssumed()) {
          return failure();
        }
        type = state.getAssumed();
      }
    }
    auto newFuncType = FunctionType::get(callableOp->getContext(),
                                         argumentTypes, newResultTypes);
    cachedFunctionTypeMap[callableOp] = newFuncType;
    return newFuncType;
  }

  // Returns the function type the callee of this caller should be converted to.
  FailureOr<FunctionType>
  lookupConvertedFunctionArgs(Operation *caller, SymbolRefAttr targetSymbol) {
    auto callableOp =
        solver.getExplorer()
            .getSymbolTables()
            .lookupNearestSymbolFrom<CallableOpInterface>(caller, targetSymbol);
    if (!callableOp) {
      return failure();
    }
    return lookupConvertedFunctionArgs(callableOp);
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  llvm::SmallDenseMap<Operation *, FunctionType> cachedFunctionTypeMap;
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

static Value castIfMismatched(OpBuilder &b, Location loc, Value v, Type t) {
  if (v.getType() != t) {
    assert(isa<MemRefType>(v.getType()) &&
           "unexpected non-memref type mismatch");
    return memref::CastOp::create(b, loc, t, v);
  }
  return v;
}

static void castIfMismatched(OpBuilder &b, Location loc,
                             MutableArrayRef<Value> vals,
                             TypeRange targetTypes) {
  for (auto [v, t] : llvm::zip_equal(vals, targetTypes)) {
    v = castIfMismatched(b, loc, v, t);
  }
}

/// Convert all result tied values of a `pcf.generic`, e.g.
///
/// ```
///   %src = ... : memref<?x?xi32, strided<[?, 1]>, 3>
///   %0 = pcf.generic
///     execute(%ref = %src : (!pcf.sref<?x?xi32>)
///                         -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
///     // some use %ref : !pcf.sref<?x?xi32, #pcf.test_scope>
///   }
/// ```
///
/// Becomes:
///
/// ```
///   %0 = pcf.generic execute {
///     // some use %src : memref<?x?xi32, strided<[?, 1]>, 3>
///   }
/// ```
///
/// Additionally the initializer region is inlined into the main body of the
/// pcf.generic as a part of this pattern as all allocations are resolved as a
/// part of this pass.
/// TODO(qedawkins): separate initializer handling out of this pass.
struct ConvertGenericOp final : OpConversionPattern<PCF::GenericOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::all_of(genericOp.getResultTypes(), llvm::IsaPred<MemRefType>)) {
      return rewriter.notifyMatchFailure(
          genericOp, "expected all parallel op results to be of memref type");
    }

    Location loc = genericOp.getLoc();
    IntegerAttr alignment =
        genericOp.getScope().getPreferredAllocAlignment(rewriter.getContext());
    SmallVector<Value> replacements;

    // Init iterator.
    auto currInit = genericOp.getInits().begin();
    ValueRange dynamicSizes = genericOp.getDynamicSizes();
    for (auto [resultType, isTied] :
         llvm::zip_equal(genericOp.getResultTypes(), genericOp.getIsTied())) {
      if (isTied) {
        replacements.push_back(*currInit);
        ++currInit;
      } else {
        int64_t numDynamicDims =
            cast<ShapedType>(resultType).getNumDynamicDims();
        replacements.push_back(memref::AllocOp::create(
            rewriter, loc, resultType, dynamicSizes.take_front(numDynamicDims),
            /*symbolOperands=*/ValueRange(), alignment));
        dynamicSizes = dynamicSizes.drop_front(numDynamicDims);
      }
    }

    // Create a new op and take the body of the current one.
    auto newGenericOp = PCF::GenericOp::create(
        rewriter, loc, genericOp.getScope(), genericOp.getNumIterators(),
        genericOp.getSyncOnReturn());
    SmallVector<Value> newArgs;
    Block *newEntry = &newGenericOp.getRegion().front();
    // By this point all globally initialized values should have been resolved.
    // Inline the initializer into the main body.
    if (!genericOp.getInitializer().empty()) {
      Block &initializerBlock = genericOp.getInitializer().front();
      auto initProducedVals =
          cast<PCF::YieldOp>(initializerBlock.getTerminator()).getOperands();
      newArgs.append(initProducedVals.begin(), initProducedVals.end());
      rewriter.eraseOp(initializerBlock.getTerminator());
      rewriter.inlineBlockBefore(&initializerBlock, newEntry,
                                 newEntry->begin());
    }
    newArgs.append(replacements);
    llvm::append_range(newArgs, newGenericOp.getRegion().getArguments());
    // Inline the entry block into the new region.
    Block *entryBlock = &genericOp.getRegion().front();
    rewriter.inlineBlockBefore(entryBlock, newEntry, newEntry->end(), newArgs);

    // Move the remaining blocks into the new region.
    for (Block &block : genericOp.getRegion()) {
      rewriter.moveBlockBefore(&block, &newGenericOp.getRegion(),
                               newGenericOp.getRegion().end());
    }

    // replace the old op.
    rewriter.replaceOp(genericOp, replacements);
    return success();
  }
};

/// Convert all result tied values of a `pcf.loop`, e.g.
///
/// ```
///   %src = ... : memref<?x?xi32, strided<[?, 1]>, 3>
///   %0 = pcf.loop
///     execute(%ref = %src : (!pcf.sref<?x?xi32>)
///                         -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
///     // some use %ref : !pcf.sref<?x?xi32, #pcf.test_scope>
///   }
/// ```
///
/// Becomes:
///
/// ```
///   %0 = pcf.loop execute {
///     // some use %src : memref<?x?xi32, strided<[?, 1]>, 3>
///   }
/// ```
struct ConvertLoopOp final : OpConversionPattern<PCF::LoopOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::LoopOp loopOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::all_of(loopOp.getResultTypes(), llvm::IsaPred<MemRefType>)) {
      return rewriter.notifyMatchFailure(
          loopOp, "expected all parallel op results to be of memref type");
    }

    Location loc = loopOp.getLoc();
    IntegerAttr alignment =
        loopOp.getScope().getPreferredAllocAlignment(rewriter.getContext());
    SmallVector<Value> replacements;

    // Init iterator.
    auto currInit = loopOp.getInits().begin();
    ValueRange dynamicSizes = loopOp.getDynamicSizes();
    for (auto [resultType, isTied] :
         llvm::zip_equal(loopOp.getResultTypes(), loopOp.getIsTied())) {
      if (isTied) {
        replacements.push_back(*currInit);
        ++currInit;
        continue;
      }

      int64_t numDynamicDims = cast<ShapedType>(resultType).getNumDynamicDims();
      replacements.push_back(memref::AllocOp::create(
          rewriter, loc, resultType, dynamicSizes.take_front(numDynamicDims),
          /*symbolOperands=*/ValueRange(), alignment));
      dynamicSizes = dynamicSizes.drop_front(numDynamicDims);
    }

    // Create a new op and take the body of the current one.
    auto newLoopOp =
        PCF::LoopOp::create(rewriter, loc, loopOp.getScope(), loopOp.getCount(),
                            loopOp.getSyncOnReturn());
    SmallVector<Value> newArgs(replacements);
    newArgs.append(newLoopOp.getRegion().getArguments().begin(),
                   newLoopOp.getRegion().getArguments().end());
    Block *entryBlock = &loopOp.getRegion().front();
    Block *newEntry = &newLoopOp.getRegion().front();
    // Inline the body into the new region.
    rewriter.inlineBlockBefore(entryBlock, newEntry, newEntry->begin(),
                               newArgs);

    // replace the old op.
    rewriter.replaceOp(loopOp, replacements);
    return success();
  }
};

/// Converts a `pcf.write_slice` to a write to the analysis yielded memref
/// destination. The flavor of write is determined by the input type:
/// 1. Tensor types are written using `iree_codegen.store_to_buffer`.
/// 2. Vector types are written using `vector.transfer_write`.
/// 3. Memref types are written with a `memref.copy`.
struct ConvertWriteSliceOp final : OpConversionPattern<PCF::WriteSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::WriteSliceOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destSlice = memref::SubViewOp::create(
        rewriter, writeOp.getLoc(), adaptor.getDest(),
        writeOp.getMixedOffsets(), writeOp.getMixedSizes(),
        writeOp.getMixedStrides());
    return llvm::TypeSwitch<Type, LogicalResult>(writeOp.getSourceType())
        .Case([&](RankedTensorType tensor) {
          rewriter.replaceOpWithNewOp<IREE::Codegen::StoreToBufferOp>(
              writeOp, writeOp.getSource(), destSlice);
          return success();
        })
        .Case([&](MemRefType memref) {
          rewriter.replaceOpWithNewOp<memref::CopyOp>(
              writeOp, writeOp.getSource(), destSlice);
          return success();
        })
        .Case([&](VectorType vector) {
          SmallVector<bool> inBounds(vector.getRank(), true);
          for (auto [inBound, vecSize, storeSize] : llvm::zip_equal(
                   inBounds, vector.getShape(), writeOp.getStaticSizes())) {
            // Since vectors must be statically sized we can just check for
            // equality here.
            inBound = vecSize == storeSize;
          }
          SmallVector<Value> offsets(
              vector.getRank(),
              arith::ConstantIndexOp::create(rewriter, writeOp.getLoc(), 0));
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, writeOp.getSource(), destSlice, offsets, inBounds);
          return success();
        })
        .Default([](Type) { return failure(); });
  }
};

/// Converts a `pcf.read_slice` to a read from the analysis yielded memref
/// source. The flavor of read is determined by the result type:
/// 1. Tensor types are read using `iree_codegen.load_from_buffer`.
/// 2. Vector types are read using `vector.transfer_read`.
struct ConvertReadSliceOp final : OpConversionPattern<PCF::ReadSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::ReadSliceOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value sourceSlice = memref::SubViewOp::create(
        rewriter, readOp.getLoc(), adaptor.getSource(),
        readOp.getMixedOffsets(), readOp.getMixedSizes(),
        readOp.getMixedStrides());
    return llvm::TypeSwitch<Type, LogicalResult>(readOp.getResultType())
        .Case([&](RankedTensorType tensor) {
          rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
              readOp, tensor, sourceSlice);
          return success();
        })
        .Case([&](VectorType vector) {
          SmallVector<bool> inBounds(vector.getRank(), true);
          for (auto [inBound, vecSize, loadSize] : llvm::zip_equal(
                   inBounds, vector.getShape(), readOp.getStaticSizes())) {
            // Since vectors must be statically sized we can just check for
            // equality here.
            inBound = vecSize == loadSize;
          }
          SmallVector<Value> offsets(
              vector.getRank(),
              arith::ConstantIndexOp::create(rewriter, readOp.getLoc(), 0));
          // Use zero padding value.
          Value zeroPadding = arith::ConstantOp::create(
              rewriter, readOp.getLoc(), vector.getElementType(),
              rewriter.getZeroAttr(vector.getElementType()));
          rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
              readOp, vector, sourceSlice, offsets, zeroPadding, inBounds);
          return success();
        })
        .Default([](Type) { return failure(); });
  }
};

/// `pcf.get_memref` always returns a maximally dynamic memref and relies on
/// propagation patterns after this pass to update everything. As a result
/// this conversion pattern simply creates a `memref.cast` +
/// `memref.memory_space_cast` + `memref.subview`.
struct ConvertGetMemrefOp final : OpConversionPattern<PCF::GetMemrefOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::GetMemrefOp getMemrefOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The source has been converted to memref with refined layout and memory
    // space. We need to cast it to match the maximally dynamic layout and no
    // memory space of the GetMemrefOp result type.
    Value source = adaptor.getSource();
    auto sourceType = cast<MemRefType>(source.getType());
    auto resultType = getMemrefOp.getResultType();

    // Cast away memory space if present.
    if (sourceType.getMemorySpace()) {
      auto noMemSpaceType =
          MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
                          sourceType.getLayout(), nullptr);
      source = memref::MemorySpaceCastOp::create(rewriter, getMemrefOp.getLoc(),
                                                 noMemSpaceType, source);
      sourceType = noMemSpaceType;
    }

    // Cast to maximally dynamic layout to match the return type.
    if (sourceType.getLayout() != resultType.getLayout()) {
      source = memref::CastOp::create(rewriter, getMemrefOp.getLoc(),
                                      resultType, source);
    }

    // Create subview using the slice params.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        getMemrefOp, source, getMemrefOp.getMixedOffsets(),
        getMemrefOp.getMixedSizes(), getMemrefOp.getMixedStrides());
    return success();
  }
};

/// Converts `pcf.alloc` to a `memref.alloc` with the requested memref type.
/// The memory space was determined during the analysis using the scope and
/// the alignment is set based on the preferred allocation alignment, also per
/// the scope.
struct ConvertAllocOp final : OpConversionPattern<PCF::AllocOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PCF::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto allocType =
        getTypeConverter()->convertType<MemRefType>(allocOp.getResult());
    if (!allocType) {
      return rewriter.notifyMatchFailure(allocOp,
                                         "failed to convert alloc type");
    }

    // TODO(qedawkins): This pattern is a hack. We should be directly allocating
    // memory as a global here. Instead we rely on dubious hoisting patterns to
    // make this work as intended.
    IntegerAttr alignment =
        allocOp.getResultType().getScope().getPreferredAllocAlignment(
            rewriter.getContext());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        allocOp, allocType, adaptor.getDynamicSizes(),
        /*symbolOperands=*/ValueRange(), alignment);
    return success();
  }
};

struct ConvertOptimizationBarrier final
    : OpConversionPattern<Util::OptimizationBarrierOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Util::OptimizationBarrierOp barrier, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Util::OptimizationBarrierOp>(
        barrier, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Control Flow Conversion Patterns
//===----------------------------------------------------------------------===//

/// The type conversions required for this pass need additional casts inserted
/// across control flow boundaries in some cases. As a result we can't use the
/// standard provided conversion patterns for the Func dialect.

/// Converts the operand and result types of the CallOp, used together with the
/// FuncOpSignatureConversion.
struct ConvertFuncOp final : OpConversionPattern<func::FuncOp> {
  ConvertFuncOp(TypeConverter &typeConverter, MLIRContext *context,
                SRefLayoutAnalysis &mapping)
      : OpConversionPattern(typeConverter, context), mapping(mapping) {}

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<FunctionType> maybeNewFunctionType =
        mapping.lookupConvertedFunctionArgs(funcOp);
    if (failed(maybeNewFunctionType)) {
      return failure();
    }
    FunctionType newFunctionType = maybeNewFunctionType.value();

    // Convert the original function types.
    TypeConverter::SignatureConversion result(newFunctionType.getNumInputs());
    for (auto [i, t] : llvm::enumerate(newFunctionType.getInputs())) {
      result.addInputs(i, t);
    }
    if (failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                           *getTypeConverter(), &result))) {
      return failure();
    }

    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newFunctionType); });
    return success();
  }

private:
  SRefLayoutAnalysis &mapping;
};

struct ConvertReturnOp final : OpConversionPattern<func::ReturnOp> {
public:
  ConvertReturnOp(TypeConverter &typeConverter, MLIRContext *context,
                  SRefLayoutAnalysis &mapping)
      : OpConversionPattern(typeConverter, context), mapping(mapping) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto parent = cast<func::FuncOp>(op->getParentOp());
    FailureOr<FunctionType> targetFuncType =
        mapping.lookupConvertedFunctionArgs(parent);
    if (failed(targetFuncType)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to get converted parent type");
    }

    SmallVector<Value> operands(adaptor.getOperands());
    castIfMismatched(rewriter, op.getLoc(), operands,
                     targetFuncType.value().getResults());
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);
    return success();
  }

private:
  SRefLayoutAnalysis &mapping;
};

/// Converts the operand and result types of the CallOp, used together with the
/// FuncOpSignatureConversion.
struct ConvertCallOp final : OpConversionPattern<func::CallOp> {
  ConvertCallOp(TypeConverter &typeConverter, MLIRContext *context,
                SRefLayoutAnalysis &mapping)
      : OpConversionPattern(typeConverter, context), mapping(mapping) {}

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    // Convert the context aware result types.
    for (Value v : callOp.getResults()) {
      Type newType = typeConverter->convertType(v);
      if (!newType) {
        return rewriter.notifyMatchFailure(callOp,
                                           "could not convert result type");
      }
      resultTypes.push_back(newType);
    }

    FailureOr<FunctionType> targetFuncType =
        mapping.lookupConvertedFunctionArgs(
            callOp, cast<SymbolRefAttr>(callOp.getCallableForCallee()));
    if (failed(targetFuncType)) {
      return rewriter.notifyMatchFailure(callOp,
                                         "could not convert argument types");
    }

    SmallVector<Value> operands(adaptor.getOperands());
    castIfMismatched(rewriter, callOp.getLoc(), operands,
                     targetFuncType.value().getInputs());

    // Substitute with the new result types from the corresponding FuncType
    // conversion.
    auto newCallOp = func::CallOp::create(
        rewriter, callOp.getLoc(), callOp.getCallee(), resultTypes, operands);
    rewriter.replaceOp(callOp, newCallOp);
    return success();
  }

private:
  SRefLayoutAnalysis &mapping;
};

//===----------------------------------------------------------------------===//
// SCF Conversion Pattern overrides
//===----------------------------------------------------------------------===//

/// The type conversions required for this pass need additional casts inserted
/// across control flow boundaries in some cases. As a result we can't use the
/// standard provided conversion patterns for the SCF dialect.

struct ConvertForOp final : OpConversionPattern<scf::ForOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    // Convert the result types.
    for (Value v : op.getResults()) {
      Type newType = typeConverter->convertType(v);
      if (!newType) {
        return rewriter.notifyMatchFailure(op, "could not convert result type");
      }
      resultTypes.push_back(newType);
    }

    SmallVector<Value> inits(adaptor.getInitArgs());
    castIfMismatched(rewriter, op.getLoc(), inits, resultTypes);

    auto newOp =
        scf::ForOp::create(rewriter, op.getLoc(), adaptor.getLowerBound(),
                           adaptor.getUpperBound(), adaptor.getStep(), inits,
                           /*bodyBuilder=*/nullptr, adaptor.getUnsignedCmp());
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter))) {
      return failure();
    }

    // Drop the rewriter created block.
    rewriter.eraseBlock(newOp.getBody(0));

    // Inline the original (now converted) body.
    auto &dstRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ConvertWhileOp final : OpConversionPattern<scf::WhileOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    // Convert the result types.
    for (Value v : op.getResults()) {
      Type newType = typeConverter->convertType(v);
      if (!newType) {
        return rewriter.notifyMatchFailure(op, "could not convert result type");
      }
      resultTypes.push_back(newType);
    }

    SmallVector<Value> inits(adaptor.getOperands());
    castIfMismatched(rewriter, op.getLoc(), inits, resultTypes);

    auto newOp =
        scf::WhileOp::create(rewriter, op.getLoc(), resultTypes, inits);
    for (auto i : {0u, 1u}) {
      if (failed(rewriter.convertRegionTypes(&op.getRegion(i), *typeConverter)))
        return failure();
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
    }

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

void ConvertSRefToMemRefPass::runOnOperation() {
  MLIRContext *context = &getContext();

  SRefLayoutAnalysis analysis(getOperation());
  if (failed(analysis.run())) {
    return signalPassFailure();
  }

  TypeConverter typeConverter;
  ConversionTarget conversionTarget(getContext());
  RewritePatternSet patterns(&getContext());

  // Add a context aware type converter that uses the layout analysis.
  typeConverter.addConversion([&](Value v) -> std::optional<Type> {
    if (isa<PCF::ShapedRefType>(v.getType())) {
      FailureOr<MemRefType> maybeConvertedType = analysis.getConvertedType(v);
      if (failed(maybeConvertedType)) {
        return Type();
      }
      return maybeConvertedType.value();
    }
    // Passthrough for everything else.
    return v.getType();
  });

  ConversionTarget target(*context);
  static constexpr auto isIllegalType = llvm::IsaPred<PCF::ShapedRefType>;

  // Verify that all operand, result, and region argument types have been
  // converted. This does not use the type converter because the type converter
  // only implements context specific conversions.
  auto isLegallyTypedOp = [&](Operation *op) -> bool {
    for (Type type : op->getResultTypes()) {
      if (isIllegalType(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type))
        return false;
    }
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes()) {
        if (isIllegalType(type))
          return false;
      }
    }
    if (auto funcInterface = dyn_cast<FunctionOpInterface>(op)) {
      if (llvm::any_of(funcInterface.getArgumentTypes(), isIllegalType)) {
        return false;
      }
      if (llvm::any_of(funcInterface.getResultTypes(), isIllegalType)) {
        return false;
      }
    }
    return true;
  };
  target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);
  ConversionConfig config;
  config.allowPatternRollback = false;

  patterns.add<ConvertGenericOp, ConvertLoopOp, ConvertWriteSliceOp,
               ConvertReadSliceOp, ConvertGetMemrefOp, ConvertAllocOp,
               ConvertOptimizationBarrier>(typeConverter, context);

  // Function related conversion patterns need the analysis to lookup function
  // type conversions.
  patterns.add<ConvertFuncOp, ConvertReturnOp, ConvertCallOp>(
      typeConverter, context, analysis);

  // Use pattern benefit to override the conversion patterns for scf.for/while
  // loops since they don't convert operand types properly. All other scf ops
  // don't branch conditionally with an operand that could be an sref and are
  // fine to leave as is.
  patterns.insert<ConvertForOp, ConvertWhileOp>(typeConverter, context,
                                                /*benefit=*/2);
  scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config))) {
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
