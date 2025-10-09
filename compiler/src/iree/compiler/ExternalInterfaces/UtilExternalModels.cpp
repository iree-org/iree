// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/UtilExternalModels.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// InferIntDivisibilityOpInterface
//===----------------------------------------------------------------------===//

static IREE::Util::ConstantIntDivisibility
getDivisibilityOfOperand(Value v,
                         IREE::Util::IntegerDivisibility divisibility) {
  if (!divisibility.isUninitialized()) {
    return divisibility.getValue();
  }
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal))) {
    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    return IREE::Util::ConstantIntDivisibility(udiv, sdiv);
  }
  return IREE::Util::ConstantIntDivisibility(1, 1);
}

struct ArithConstantInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithConstantInferIntDivisibilityOpInterface, arith::ConstantOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto constOp = cast<arith::ConstantOp>(op);
    auto constAttr = dyn_cast_if_present<IntegerAttr>(constOp.getValue());
    if (constAttr) {
      const APInt &value = constAttr.getValue();
      uint64_t udiv = value.getZExtValue();
      uint64_t sdiv = std::abs(value.getSExtValue());
      setResultDivs(constOp.getResult(),
                    IREE::Util::ConstantIntDivisibility(udiv, sdiv));
    }
  }
};

struct ArithMulIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithMulIInferIntDivisibilityOpInterface, arith::MulIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto mulOp = cast<arith::MulIOp>(op);

    auto lhsDivisibility = getDivisibilityOfOperand(mulOp.getLhs(), argDivs[0]);
    auto rhsDivisibility = getDivisibilityOfOperand(mulOp.getRhs(), argDivs[1]);

    uint64_t mulUDiv = lhsDivisibility.udiv() * rhsDivisibility.udiv();
    uint64_t mulSDiv = lhsDivisibility.sdiv() * rhsDivisibility.sdiv();

    setResultDivs(mulOp.getResult(),
                  IREE::Util::ConstantIntDivisibility(mulUDiv, mulSDiv));
  }
};

struct ArithDivUIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithDivUIInferIntDivisibilityOpInterface, arith::DivUIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto divOp = cast<arith::DivUIOp>(op);

    APInt intVal;
    if (!matchPattern(divOp.getRhs(), m_ConstantInt(&intVal))) {
      return;
    }

    auto lhsDivisibility = getDivisibilityOfOperand(divOp.getLhs(), argDivs[0]);

    uint64_t divUDiv = lhsDivisibility.udiv() / intVal.getZExtValue();
    uint64_t divSDiv = lhsDivisibility.sdiv() / std::abs(intVal.getSExtValue());

    setResultDivs(divOp, IREE::Util::ConstantIntDivisibility(divUDiv, divSDiv));
  }
};

//===----------------------------------------------------------------------===//
// ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

/// For some reason, this interface has to be done as an external model.
struct UtilAssumeIntValueBoundsOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          UtilAssumeIntValueBoundsOpInterface, IREE::Util::AssumeIntOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto assumeOp = cast<IREE::Util::AssumeIntOp>(op);
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "value is a result of this index op");
    auto [min, max] =
        assumeOp.getUnionedUnsignedRange(result.getResultNumber());

    std::optional<int64_t> udiv =
        assumeOp.getUnionedUnsignedDivisor(result.getResultNumber());

    if (min) {
      cstr.bound(result) >= *min;
    }
    if (max) {
      cstr.bound(result) <= *max;
    }
    if (udiv) {
      // To represent the divisibility guarantee, emit a bound clamping the
      // value to the udiv value. i.e.
      //
      // v == floordiv(v, udiv) * udiv
      //
      // Mod/divide folders can cleanup such terms with the appropriate bounds
      // query.
      AffineExpr expr =
          cstr.getExpr(assumeOp.getOperand(result.getResultNumber()));
      AffineExpr udivCst =
          getAffineConstantExpr(udiv.value(), op->getContext());
      AffineExpr clampExpr = expr.floorDiv(udivCst) * udivCst;
      cstr.bound(result) == clampExpr;
    }
  }
};

//===----------------------------------------------------------------------===//
// GlobalOpInterface
//===----------------------------------------------------------------------===//

struct GlobalOpInterfaceExternalModel
    : public IREE::Util::GlobalOpInterface::ExternalModel<
          GlobalOpInterfaceExternalModel, ml_program::GlobalOp> {
  Attribute getGlobalInitialValue(Operation *op) const {
    return cast<ml_program::GlobalOp>(op).getValueAttr();
  }
  void setGlobalInitialValue(Operation *op, Attribute value) const {
    if (value) {
      cast<ml_program::GlobalOp>(op).setValueAttr(value);
    } else {
      cast<ml_program::GlobalOp>(op).removeValueAttr();
    }
  }

  IREE::Util::InliningPolicyAttrInterface
  getGlobalInliningPolicy(Operation *op) const {
    if (op->hasAttr("noinline"))
      return IREE::Util::InlineNeverAttr::get(op->getContext());
    return {};
  }
  void
  setGlobalInliningPolicy(Operation *op,
                          IREE::Util::InliningPolicyAttrInterface value) const {
    if (isa_and_nonnull<IREE::Util::InlineNeverAttr>(value)) {
      op->setAttr("noinline", UnitAttr::get(op->getContext()));
    } else {
      op->removeAttr("noinline");
    }
  }

  IREE::Util::GlobalLoadOpInterface createLoadOp(Operation *op, Location loc,
                                                 OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    if (globalOp.getIsMutable()) {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          ml_program::GlobalLoadOp::create(builder, loc, globalOp.getType(),
                                           FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    } else {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          ml_program::GlobalLoadConstOp::create(
              builder, loc, globalOp.getType(),
              FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    }
  }

  IREE::Util::GlobalStoreOpInterface createStoreOp(Operation *op, Location loc,
                                                   Value value,
                                                   OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    return cast<IREE::Util::GlobalStoreOpInterface>(
        ml_program::GlobalStoreOp ::create(
            builder, loc, FlatSymbolRefAttr::get(globalOp), value)
            .getOperation());
  }
};

//===----------------------------------------------------------------------===//
// NumericCastOpInterface
//===----------------------------------------------------------------------===//

// Since all details of the interface are provided via default implementations,
// we can just have one templated external model to apply per op, vs one
// explicit model per op.
struct GenericNumericCastExternalModel {
  template <typename OpTy>
  struct ExternalModel
      : public IREE::Util::NumericCastOpInterface::ExternalModel<
            ExternalModel<OpTy>, OpTy> {};

  template <typename OpTy>
  static void add(MLIRContext *context) {
    OpTy::template attachInterface<ExternalModel<OpTy>>(*context);
  }

  template <typename OpTy1, typename OpTy2, typename... More>
  static void add(MLIRContext *context) {
    add<OpTy1>(context);
    add<OpTy2, More...>(context);
  }
};

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

struct InsertSliceOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          InsertSliceOpTiedOpInterface, tensor::InsertSliceOp> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        insertSliceOp.getDest());
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    return {1}; // dest
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    return {1}; // dest
  }
};

template <typename OpTy>
struct LinalgOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          LinalgOpTiedOpInterface<OpTy>, OpTy> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        linalgOp.getDpsInits()[resultIndex]);
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return {linalgOp.getDpsInitsMutable()[resultIndex].getOperandNumber()};
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    SmallVector<int64_t> result;
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      result.push_back(*getTiedResultOperandIndex(op, i));
    return result;
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `TiedOpInterface` with each of them.
template <typename... Ops>
struct LinalgOpTiedOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (void)std::initializer_list<int>{
        0,
        (Ops::template attachInterface<LinalgOpTiedOpInterface<Ops>>(*context),
         0)...};
  }
};

//===----------------------------------------------------------------------===//
// HoistableOpInterface
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct UnhoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          UnhoistableOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return false; }
  bool isHoistableLeafOp(Operation *) const { return false; }
};

template <typename OpTy>
struct HoistableNonLeafOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableNonLeafOpInterface<OpTy>, OpTy> {
  bool isHoistableLeafOp(Operation *) const { return false; }
};

// The default interface is always hoistable. This acts as an override
// for other default hoistability checks as the interface is checked
// first.
template <typename OpTy>
struct AlwaysHoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          AlwaysHoistableOpInterface<OpTy>, OpTy> {};

template <typename OpTy>
struct HoistableLinalgOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableLinalgOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return true; }

  // Determines if a linalg op is a hoistable leaf, based on heuristics.
  bool isHoistableLeafOp(Operation *op) const {
    // Don't hoist bit extend ops because fusing them with their
    // consumers prevents materializing the high bit-width tensor and they
    // preform very little real computation.
    if (IREE::LinalgExt::isBitExtendOp(op)) {
      return false;
    }

    // Hoist all non-generic linalg ops except for fill ops which should be
    // fused with their consumers.
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp) {
      return !isa<linalg::FillOp>(op);
    }

    // Don't hoist ops with no tensor inputs. They are likely to be fill-like
    // or sequences (from `linalg.index`) which can be fused with their
    // consumers.
    if (IREE::LinalgExt::hasOnlyScalarInputs(genericOp)) {
      return false;
    }

    // Don't hoist broadcast-like ops because fusing them makes the new
    // op cheaper.
    if (linalg::isaBroadcastOpInterface(genericOp).has_value()) {
      return false;
    }

    // Hoist all other ops.
    return true;
  }
  bool isAtomicallyHoistableOp(Operation *) const { return true; }
  bool isOperandHoistable(Operation *, OpOperand *) const { return true; }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated Hoistable___OpInterface.
template <typename... Ops>
struct UnhoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<UnhoistableOpInterface<Ops>>(*context), ...);
  }
};

template <typename... Ops>
struct HoistableNonLeafOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableNonLeafOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct AlwaysHoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<AlwaysHoistableOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct HoistableLinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableLinalgOpInterface<Ops>>(*context),
     ...);
  }
};

//===----------------------------------------------------------------------===//
// MutableRegionBranchOpInterface
//===----------------------------------------------------------------------===//

// External model for scf.for operation.
struct SCFForOpMutableRegionBranchOpInterface
    : public IREE::Util::MutableRegionBranchOpInterface::ExternalModel<
          SCFForOpMutableRegionBranchOpInterface, scf::ForOp> {
  Operation *rebuildWithExpandedTypes(
      Operation *op,
      llvm::function_ref<void(Type, SmallVectorImpl<Type> &)> expandTypeFn,
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &, OpBuilder &)>
          expandOperandFn,
      std::optional<
          llvm::function_ref<Value(Location, Value, ValueRange, OpBuilder &)>>
          wrapExpandedBlockArgFn,
      llvm::function_ref<void(Region &, bool /*canModifyEntryBlock*/)>
          expandRegionFn,
      OpBuilder &builder) const {
    auto forOp = cast<scf::ForOp>(op);
    Location loc = forOp.getLoc();

    // Expand iter_args operands (lb, ub, step are never expanded).
    SmallVector<Value> newIterArgs;
    for (Value iterArg : forOp.getInitArgs()) {
      expandOperandFn(iterArg, newIterArgs, builder);
    }

    // Expand result types.
    SmallVector<Type> newResultTypes;
    for (Type resultType : forOp.getResultTypes()) {
      expandTypeFn(resultType, newResultTypes);
    }

    // Create new for loop with expanded signature.
    auto newForOp =
        scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newIterArgs);

    // Move the body from old to new.
    newForOp.getBodyRegion().takeBody(forOp.getBodyRegion());

    // The new op now has the old body but with unexpanded block arguments.
    // We need to update the block arguments to match the expanded iter args.
    Block &body = *newForOp.getBody();

    // Insert new block arguments for expanded types.
    // Start from the back to preserve indices.
    // Note we skip the induction var at 0.
    for (int i = body.getNumArguments() - 1; i >= 1; --i) {
      auto arg = body.getArgument(i);
      SmallVector<Type> expandedTypes;
      expandTypeFn(arg.getType(), expandedTypes);
      if (expandedTypes.size() > 1) {
        // This type expands to multiple values.
        // Insert new arguments after the current one.
        SmallVector<Value> expandedArgs;
        expandedArgs.push_back(arg);
        for (unsigned j = 1; j < expandedTypes.size(); ++j) {
          expandedArgs.push_back(
              body.insertArgument(i + j, expandedTypes[j], arg.getLoc()));
        }

        // If wrapper callback provided, create replacement value and update
        // uses.
        if (wrapExpandedBlockArgFn) {
          OpBuilder blockBuilder(&body, body.begin());
          Value replacement = (*wrapExpandedBlockArgFn)(
              arg.getLoc(), arg, expandedArgs, blockBuilder);
          arg.replaceAllUsesExcept(replacement, replacement.getDefiningOp());
        }
      }
    }

    // Recursively expand operations in the body.
    expandRegionFn(newForOp.getBodyRegion(), /*canModifyEntryBlock=*/false);

    // Update the yield to match expanded result types.
    if (body.mightHaveTerminator()) {
      auto *terminator = body.getTerminator();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        SmallVector<Value> newYieldOperands;
        builder.setInsertionPoint(yieldOp);
        for (Value operand : yieldOp.getOperands()) {
          expandOperandFn(operand, newYieldOperands, builder);
        }
        scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldOperands);
        yieldOp.erase();
      }
    }

    return newForOp;
  }

  llvm::BitVector getPreservedBlockArguments(Operation *op,
                                             unsigned regionIndex) const {
    auto forOp = cast<scf::ForOp>(op);
    llvm::BitVector preserved(forOp.getBody()->getNumArguments());
    preserved.set(0); // Preserve induction variable.
    return preserved;
  }

  OperandRange getRegionEntryOperands(Operation *op,
                                      unsigned regionIndex) const {
    auto forOp = cast<scf::ForOp>(op);
    return forOp.getInitArgs();
  }

  ResultRange getRegionExitResults(Operation *op, unsigned regionIndex) const {
    return op->getResults();
  }

  OperandRange getExpandableTerminatorOperands(Operation *op,
                                               Operation *terminator,
                                               unsigned regionIndex) const {
    // For scf.yield, all operands are expandable.
    return terminator->getOperands();
  }
};

// External model for scf.if operation.
struct SCFIfOpMutableRegionBranchOpInterface
    : public IREE::Util::MutableRegionBranchOpInterface::ExternalModel<
          SCFIfOpMutableRegionBranchOpInterface, scf::IfOp> {
  Operation *rebuildWithExpandedTypes(
      Operation *op,
      llvm::function_ref<void(Type, SmallVectorImpl<Type> &)> expandTypeFn,
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &, OpBuilder &)>
          expandOperandFn,
      std::optional<
          llvm::function_ref<Value(Location, Value, ValueRange, OpBuilder &)>>
          wrapExpandedBlockArgFn,
      llvm::function_ref<void(Region &, bool /*canModifyEntryBlock*/)>
          expandRegionFn,
      OpBuilder &builder) const {
    // Note: scf.if has no block arguments, so wrapExpandedBlockArgFn is unused.
    (void)wrapExpandedBlockArgFn;
    auto ifOp = cast<scf::IfOp>(op);
    Location loc = ifOp.getLoc();

    // Expand result types.
    SmallVector<Type> newResultTypes;
    for (Type resultType : ifOp.getResultTypes()) {
      expandTypeFn(resultType, newResultTypes);
    }

    // Create new if op with expanded result types.
    auto newIfOp =
        scf::IfOp::create(builder, loc, newResultTypes, ifOp.getCondition(),
                          /*withElseRegion=*/!ifOp.getElseRegion().empty());

    // Copy regions from old to new op.
    // Use takeBody to move the region contents.
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty()) {
      newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    }

    // Now recursively expand operations in the regions.
    // This will handle the expansion of operations inside.
    expandRegionFn(newIfOp.getThenRegion(), /*canModifyEntryBlock=*/false);
    if (!newIfOp.getElseRegion().empty()) {
      expandRegionFn(newIfOp.getElseRegion(), /*canModifyEntryBlock=*/false);
    }

    // Now we need to update the yield operations to match the expanded result
    // types. The yield ops need to yield expanded values for resource results.
    auto updateYield = [&](Block &block) {
      if (!block.mightHaveTerminator()) {
        return;
      }
      auto *terminator = block.getTerminator();
      if (!terminator) {
        return;
      }
      auto yieldOp = dyn_cast<scf::YieldOp>(terminator);
      if (!yieldOp) {
        return;
      }

      SmallVector<Value> newYieldOperands;
      builder.setInsertionPoint(yieldOp);
      for (Value operand : yieldOp.getOperands()) {
        expandOperandFn(operand, newYieldOperands, builder);
      }
      scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldOperands);
      yieldOp.erase();
    };

    updateYield(newIfOp.getThenRegion().front());
    if (!newIfOp.getElseRegion().empty()) {
      updateYield(newIfOp.getElseRegion().front());
    }

    return newIfOp;
  }

  llvm::BitVector getPreservedBlockArguments(Operation *op,
                                             unsigned regionIndex) const {
    // scf.if has no block arguments.
    return llvm::BitVector(0);
  }

  OperandRange getRegionEntryOperands(Operation *op,
                                      unsigned regionIndex) const {
    // No operands map to block args.
    return OperandRange(op->operand_end(), op->operand_end());
  }

  ResultRange getRegionExitResults(Operation *op, unsigned regionIndex) const {
    return op->getResults();
  }

  OperandRange getExpandableTerminatorOperands(Operation *op,
                                               Operation *terminator,
                                               unsigned regionIndex) const {
    // For scf.yield, all operands are expandable.
    return terminator->getOperands();
  }
};

// External model for scf.while operation.
struct SCFWhileOpMutableRegionBranchOpInterface
    : public IREE::Util::MutableRegionBranchOpInterface::ExternalModel<
          SCFWhileOpMutableRegionBranchOpInterface, scf::WhileOp> {
  Operation *rebuildWithExpandedTypes(
      Operation *op,
      llvm::function_ref<void(Type, SmallVectorImpl<Type> &)> expandTypeFn,
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &, OpBuilder &)>
          expandOperandFn,
      std::optional<
          llvm::function_ref<Value(Location, Value, ValueRange, OpBuilder &)>>
          wrapExpandedBlockArgFn,
      llvm::function_ref<void(Region &, bool /*canModifyEntryBlock*/)>
          expandRegionFn,
      OpBuilder &builder) const {
    auto whileOp = cast<scf::WhileOp>(op);
    Location loc = whileOp.getLoc();

    // Expand initial operands.
    SmallVector<Value> newInits;
    for (Value init : whileOp.getInits()) {
      expandOperandFn(init, newInits, builder);
    }

    // Expand result types.
    SmallVector<Type> newResultTypes;
    for (Type resultType : whileOp.getResultTypes()) {
      expandTypeFn(resultType, newResultTypes);
    }

    // scf.while builder needs the before and after region builders.
    // We'll create it without them and then manually populate.
    auto newWhileOp =
        scf::WhileOp::create(builder, loc, newResultTypes, newInits,
                             /*beforeBuilder=*/nullptr,
                             /*afterBuilder=*/nullptr);

    // Move the regions from old to new.
    newWhileOp.getBefore().takeBody(whileOp.getBefore());
    newWhileOp.getAfter().takeBody(whileOp.getAfter());

    // Helper lambda to expand block arguments and call wrapper.
    auto expandBlockArgs = [&](Block &block) {
      for (int i = block.getNumArguments() - 1; i >= 0; --i) {
        auto arg = block.getArgument(i);
        SmallVector<Type> expandedTypes;
        expandTypeFn(arg.getType(), expandedTypes);
        if (expandedTypes.size() > 1) {
          // This type expands to multiple values.
          // Insert new arguments after the current one.
          SmallVector<Value> expandedArgs;
          expandedArgs.push_back(arg);
          for (unsigned j = 1; j < expandedTypes.size(); ++j) {
            expandedArgs.push_back(
                block.insertArgument(i + j, expandedTypes[j], arg.getLoc()));
          }

          // If wrapper callback provided, create replacement value and update
          // uses.
          if (wrapExpandedBlockArgFn) {
            OpBuilder blockBuilder(&block, block.begin());
            Value replacement = (*wrapExpandedBlockArgFn)(
                arg.getLoc(), arg, expandedArgs, blockBuilder);
            arg.replaceAllUsesExcept(replacement, replacement.getDefiningOp());
          }
        }
      }
    };

    // Expand block arguments in both regions.
    Block &beforeBlock = newWhileOp.getBefore().front();
    expandBlockArgs(beforeBlock);

    Block &afterBlock = newWhileOp.getAfter().front();
    expandBlockArgs(afterBlock);

    // Recursively expand operations in both regions.
    expandRegionFn(newWhileOp.getBefore(), /*canModifyEntryBlock=*/false);
    expandRegionFn(newWhileOp.getAfter(), /*canModifyEntryBlock=*/false);

    // Rebuild the condition with expanded operands.
    if (beforeBlock.mightHaveTerminator()) {
      if (auto condOp =
              dyn_cast<scf::ConditionOp>(beforeBlock.getTerminator())) {
        SmallVector<Value> newCondArgs;
        builder.setInsertionPoint(condOp);
        for (Value operand : condOp.getArgs()) {
          expandOperandFn(operand, newCondArgs, builder);
        }
        scf::ConditionOp::create(builder, condOp.getLoc(),
                                 condOp.getCondition(), newCondArgs);
        condOp.erase();
      }
    }

    // Rebuild the yield in the after region with expanded operands.
    if (afterBlock.mightHaveTerminator()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock.getTerminator())) {
        SmallVector<Value> newYieldOperands;
        builder.setInsertionPoint(yieldOp);
        for (Value operand : yieldOp.getOperands()) {
          expandOperandFn(operand, newYieldOperands, builder);
        }
        scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldOperands);
        yieldOp.erase();
      }
    }

    return newWhileOp;
  }

  llvm::BitVector getPreservedBlockArguments(Operation *op,
                                             unsigned regionIndex) const {
    // All block arguments in scf.while can be expanded.
    auto whileOp = cast<scf::WhileOp>(op);
    Region &region =
        regionIndex == 0 ? whileOp.getBefore() : whileOp.getAfter();
    return llvm::BitVector(region.front().getNumArguments());
  }

  OperandRange getRegionEntryOperands(Operation *op,
                                      unsigned regionIndex) const {
    auto whileOp = cast<scf::WhileOp>(op);
    if (regionIndex == 0) {
      // Before region gets the init operands.
      return whileOp.getInits();
    }
    // After region receives operands from the condition region's scf.condition
    // op, which are not direct operands of the scf.while itself. Return empty
    // range.
    return OperandRange(op->operand_end(), op->operand_end());
  }

  ResultRange getRegionExitResults(Operation *op, unsigned regionIndex) const {
    if (regionIndex == 1) {
      // After region produces the results.
      return op->getResults();
    }
    // Before region doesn't directly produce results.
    return ResultRange(op->result_end(), op->result_end());
  }

  OperandRange getExpandableTerminatorOperands(Operation *op,
                                               Operation *terminator,
                                               unsigned regionIndex) const {
    if (auto condOp = dyn_cast<scf::ConditionOp>(terminator)) {
      // For condition, expand the args but not the condition boolean.
      return condOp.getArgs();
    }
    // For scf.yield, all operands are expandable.
    return terminator->getOperands();
  }
};

// External model for scf.index_switch operation.
struct SCFIndexSwitchOpMutableRegionBranchOpInterface
    : public IREE::Util::MutableRegionBranchOpInterface::ExternalModel<
          SCFIndexSwitchOpMutableRegionBranchOpInterface, scf::IndexSwitchOp> {
  Operation *rebuildWithExpandedTypes(
      Operation *op,
      llvm::function_ref<void(Type, SmallVectorImpl<Type> &)> expandTypeFn,
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &, OpBuilder &)>
          expandOperandFn,
      std::optional<
          llvm::function_ref<Value(Location, Value, ValueRange, OpBuilder &)>>
          wrapExpandedBlockArgFn,
      llvm::function_ref<void(Region &, bool /*canModifyEntryBlock*/)>
          expandRegionFn,
      OpBuilder &builder) const {
    // Note: scf.index_switch has no block arguments, so wrapExpandedBlockArgFn
    // is unused.
    (void)wrapExpandedBlockArgFn;
    auto switchOp = cast<scf::IndexSwitchOp>(op);
    Location loc = switchOp.getLoc();

    // Expand result types.
    SmallVector<Type> newResultTypes;
    for (Type resultType : switchOp.getResultTypes()) {
      expandTypeFn(resultType, newResultTypes);
    }

    // Create new index_switch op with expanded result types.
    auto newSwitchOp = scf::IndexSwitchOp::create(
        builder, loc, newResultTypes, switchOp.getArg(), switchOp.getCases(),
        switchOp.getCases().size());

    // Clone each case region.
    for (unsigned i = 0; i < switchOp.getCaseRegions().size(); ++i) {
      Region &oldRegion = switchOp.getCaseRegions()[i];
      Region &newRegion = newSwitchOp.getCaseRegions()[i];
      if (oldRegion.empty()) {
        continue;
      }
      builder.createBlock(&newRegion);
      builder.setInsertionPointToStart(&newRegion.front());

      IRMapping mapping;
      for (Operation &op : oldRegion.front()) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
          SmallVector<Value> newYieldOperands;
          for (Value operand : yieldOp.getOperands()) {
            if (mapping.contains(operand)) {
              expandOperandFn(mapping.lookup(operand), newYieldOperands,
                              builder);
            } else {
              expandOperandFn(operand, newYieldOperands, builder);
            }
          }
          scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldOperands);
        } else {
          builder.clone(op, mapping);
        }
      }
    }

    // Clone default region.
    Region &oldDefault = switchOp.getDefaultRegion();
    Region &newDefault = newSwitchOp.getDefaultRegion();
    if (!oldDefault.empty()) {
      builder.createBlock(&newDefault);
      builder.setInsertionPointToStart(&newDefault.front());

      IRMapping mapping;
      for (Operation &op : oldDefault.front()) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
          SmallVector<Value> newYieldOperands;
          for (Value operand : yieldOp.getOperands()) {
            if (mapping.contains(operand)) {
              expandOperandFn(mapping.lookup(operand), newYieldOperands,
                              builder);
            } else {
              expandOperandFn(operand, newYieldOperands, builder);
            }
          }
          scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldOperands);
        } else {
          builder.clone(op, mapping);
        }
      }
    }

    // Recursively expand operations in all regions.
    for (Region &region : newSwitchOp.getCaseRegions()) {
      if (!region.empty()) {
        expandRegionFn(region, /*canModifyEntryBlock=*/false);
      }
    }
    if (!newSwitchOp.getDefaultRegion().empty()) {
      expandRegionFn(newSwitchOp.getDefaultRegion(),
                     /*canModifyEntryBlock=*/false);
    }

    return newSwitchOp;
  }

  llvm::BitVector getPreservedBlockArguments(Operation *op,
                                             unsigned regionIndex) const {
    // scf.index_switch has no block arguments.
    return llvm::BitVector(0);
  }

  OperandRange getRegionEntryOperands(Operation *op,
                                      unsigned regionIndex) const {
    // No operands map to block args.
    return OperandRange(op->operand_end(), op->operand_end());
  }

  ResultRange getRegionExitResults(Operation *op, unsigned regionIndex) const {
    return op->getResults();
  }

  OperandRange getExpandableTerminatorOperands(Operation *op,
                                               Operation *terminator,
                                               unsigned regionIndex) const {
    // For scf.yield, all operands are expandable.
    return terminator->getOperands();
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<ml_program::MLProgramDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();

  registry.addExtension(
      +[](MLIRContext *context, ml_program::MLProgramDialect *dialect) {
        ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(
            *context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            arith::ArithDialect *dialect) {
    GenericNumericCastExternalModel::add<
        arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
        arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp, arith::TruncFOp,
        arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp>(context);
    arith::ConstantOp::attachInterface<
        ArithConstantInferIntDivisibilityOpInterface>(*context);
    arith::MulIOp::attachInterface<ArithMulIInferIntDivisibilityOpInterface>(
        *context);
    arith::DivUIOp::attachInterface<ArithDivUIInferIntDivisibilityOpInterface>(
        *context);
  });

  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        tensor::InsertSliceOp::attachInterface<InsertSliceOpTiedOpInterface>(
            *context);
      });

  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `TiedOpInterface` to all ops
        // one-by-one.
        LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
        >::registerOpInterface(context);
  });

  // Hoistable Op Interface registration.

  // Register hoistable op interfaces for Encoding ops.
  registry.addExtension(
      +[](MLIRContext *context, IREE::Encoding::IREEEncodingDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::Encoding::SetEncodingOp>::registerOpInterface(context);
      });

  // Register hoistable op interfaces for Flow ops.
  registry.addExtension(
      +[](MLIRContext *context, IREE::Flow::FlowDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::Flow::DispatchWorkgroupCountOp>::registerOpInterface(context);

        AlwaysHoistableOpInterfaceHelper<
            IREE::Flow::TensorEncodeOp>::registerOpInterface(context);
      });

  // Register hoistable op interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Structured op implementations and a handful of pure ops are included.
        // Notably: IndexOp is not included because it establishes a hidden
        // dependency to the iterator and is non-const.

        // Register all LinalgOps ops. `LinalgOp` is an interface and it is
        // not possible to attach an external interface to an existing
        // interface. Therefore, attach the `HoistableLinalgOpInterface` to all
        // ops one-by-one.
        HoistableLinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
        UnhoistableOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
            >::registerOpInterface(context);

        AlwaysHoistableOpInterfaceHelper<
            linalg::PackOp, linalg::UnPackOp>::registerOpInterface(context);
      });
  // Register hoistable op interfaces for tensor ops.
  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        // Never hoist empty and other pure metadata ops as a leaf. It's fine to
        // hoist them as a part of a larger constant tree that does actual work.
        HoistableNonLeafOpInterfaceHelper<
            tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ExtractSliceOp>::registerOpInterface(context);
        // Cases of trivial pack/unpack should be handled as canonicalizations
        // before we get here, thus we're safe to always hoist.
        AlwaysHoistableOpInterfaceHelper<tensor::PadOp>::registerOpInterface(
            context);
      });
  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        IREE::Util::AssumeIntOp::attachInterface<
            UtilAssumeIntValueBoundsOpInterface>(*context);
      });

  // Register MutableRegionBranchOpInterface for SCF ops.
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<SCFForOpMutableRegionBranchOpInterface>(
        *context);
    scf::IfOp::attachInterface<SCFIfOpMutableRegionBranchOpInterface>(*context);
    scf::WhileOp::attachInterface<SCFWhileOpMutableRegionBranchOpInterface>(
        *context);
    scf::IndexSwitchOp::attachInterface<
        SCFIndexSwitchOpMutableRegionBranchOpInterface>(*context);
  });
}

} // namespace mlir::iree_compiler
