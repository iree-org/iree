// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SlowDynamicAPInt.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "iree-codegen-specialize-exports"

using SpecializationRangesAttrHelper = mlir::iree_compiler::IREE::Codegen::
    IREECodegenDialect::SpecializationRangesAttrHelper;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPECIALIZEEXPORTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct AssumedWorkloadSize {
  int64_t staticSize = ShapedType::kDynamic;
  int64_t workloadOrdinal;
  OpResult assumptionOrOrdinal;
};
} // namespace

/// Returns a list of sizes, or the tied workload ordinal if dynamic, for the
/// iteration domain of the target tilable op. This is needed when constructing
/// the condition region. Also returns the `util.int.assume` for each size if
/// present so we can simply update the range there instead of constructing a
/// new one.
static FailureOr<SmallVector<AssumedWorkloadSize>>
getIterationDomainAsWorkload(TilingInterface specializationRoot) {
  OpBuilder b(specializationRoot);
  SmallVector<mlir::Range> iterationSpace =
      specializationRoot.getIterationDomain(b);

  SmallVector<AssumedWorkloadSize> workloadAssumptions(iterationSpace.size());

  for (auto [i, range] : llvm::enumerate(iterationSpace)) {
    // Non-zero offset and non-unit stride unsupported.
    if (!isZeroInteger(range.offset) || !isOneInteger(range.stride)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to get zero offset + unit stride.");
      return failure();
    }

    std::optional<int64_t> constantSize = getConstantIntValue(range.size);
    if (constantSize) {
      workloadAssumptions[i].staticSize = constantSize.value();
      continue;
    }

    // Look for the ordinal defining the size. This relies on folders kicking in
    // to remove the cruft when querying for the iteration domain.
    auto size = cast<Value>(range.size);
    if (auto dimOp = size.getDefiningOp<tensor::DimOp>()) {
      std::optional<int64_t> dim = getConstantIntValue(dimOp.getDimension());
      if (dim) {
        // Walk up the SSA chain for the definition of the dynamic dim.
        size = getValueOrCreateConstantIndexOp(
            b, specializationRoot.getLoc(),
            IREE::Util::findDim(dimOp.getSource(), dim.value()));
      }
    }
    auto workloadOrdinal =
        size.getDefiningOp<IREE::TensorExt::DispatchWorkloadOrdinalOp>();
    if (!workloadOrdinal) {
      // If we can't map back to the workload, we can't build the selection
      // condition.
      LLVM_DEBUG({
        llvm::dbgs() << "Could not find workload ordinal for size\n\t";
        llvm::dbgs() << range.size << "\n";
      });
      return failure();
    }

    workloadAssumptions[i].workloadOrdinal =
        workloadOrdinal.getOrdinal().getZExtValue();

    if (!workloadOrdinal.getOperand()
             .getDefiningOp<IREE::Util::AssumeIntOp>()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Could not find int.assume for size\n\t";
        llvm::dbgs() << range.size << "\n";
      });
      // If no assume is found, point to the result of the ordinal. Later we
      // will generate an assume on the input.
      workloadAssumptions[i].assumptionOrOrdinal =
          workloadOrdinal->getOpResult(0);
    } else {
      workloadAssumptions[i].assumptionOrOrdinal =
          cast<OpResult>(workloadOrdinal.getOperand());
    }
  }
  return workloadAssumptions;
}

/// Gets the next value not present in |ordinals| that is >= |ordinal| and
/// adds it to the set.
static int64_t
getAndInsertNextAvailableOrdinal(llvm::SmallDenseSet<int64_t> &ordinals,
                                 int64_t ordinal) {
  int64_t newOrdinal = ordinal + 1;
  while (ordinals.contains(newOrdinal)) {
    ++newOrdinal;
  }
  ordinals.insert(newOrdinal);
  return newOrdinal;
}

/// Specializes the function |func| exported by |exportOp| based on
/// |specializationRoot|. The ranges of the iteration space of the tilable
/// root to specialize for is specified by |specializationRanges|.
static void specializeExportedFunction(
    IREE::HAL::ExecutableExportOp exportOp, func::FuncOp func,
    TilingInterface specializationRoot,
    IREE::Util::MultiIntAssumptionArrayAttr specializationRanges,
    llvm::SmallDenseSet<int64_t> &ordinals) {
  if (specializationRanges.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Empty specialization ranges.");
    return;
  }

  // Bail out if the export already has a fallback, or if there is no workgroup
  // count body.
  if (!exportOp.getWorkgroupCountBody() || exportOp.getConditionFallback()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Missing workgroup count body or already has fallback.");
    return;
  }

  FailureOr<SmallVector<AssumedWorkloadSize>> maybeWorkloadMapping =
      getIterationDomainAsWorkload(specializationRoot);
  if (failed(maybeWorkloadMapping)) {
    LLVM_DEBUG(llvm::dbgs() << "Empty specialization ranges.");
    return;
  }

  SymbolTable innerModule = SymbolTable::getNearestSymbolTable(func);
  SymbolTable symbolTable(innerModule);

  IREE::HAL::ExecutableExportOp currentExport = exportOp;
  func::FuncOp currentFunction = func;
  OpBuilder builder(exportOp);
  Location loc = exportOp.getLoc();

  ArrayRef<AssumedWorkloadSize> workloadMapping = maybeWorkloadMapping.value();
  for (auto specializationRange : specializationRanges) {
    [[maybe_unused]] bool requiresSpecialization = false;
    bool neverApplies = false;
    bool alwaysApplies = true;

    // First determine whether the target specialization range is even capable
    // of applying, or if it always applies.
    //
    // Use zip for implicit truncation behavior to the number of ranges. Note
    // that this will skip any ranges longer than the workload rank, potentially
    // creating unexpected successes, however the alternative is to fail.
    for (auto [range, assumedSize] :
         llvm::zip(specializationRange, workloadMapping)) {
      int64_t umin = range.getUmin().value_or(0);
      int64_t umax = range.getUmax().value_or(INT64_MAX);
      int64_t udiv = range.getUdiv().value_or(1);

      int64_t valueUmin = 0;
      int64_t valueUmax = INT64_MAX;
      int64_t valueUdiv = 1;
      if (!ShapedType::isDynamic(assumedSize.staticSize)) {
        valueUmin = assumedSize.staticSize;
        valueUmax = assumedSize.staticSize;
        valueUdiv = assumedSize.staticSize;
      } else {
        // Specialization is only needed if there is at least one dynamic shape
        // present.
        requiresSpecialization = true;

        // Infer the range/divisor of the dim based on the tied assumption.
        if (auto assumeOp = llvm::dyn_cast<IREE::Util::AssumeIntOp>(
                assumedSize.assumptionOrOrdinal.getOwner())) {
          std::pair<std::optional<int64_t>, std::optional<int64_t>>
              dynamicRange = assumeOp.getUnionedUnsignedRange(
                  assumedSize.assumptionOrOrdinal.getResultNumber());
          valueUmin = dynamicRange.first.value_or(0);
          valueUmax = dynamicRange.second.value_or(INT64_MAX);
          valueUdiv = assumeOp
                          .getUnionedUnsignedDivisor(
                              assumedSize.assumptionOrOrdinal.getResultNumber())
                          .value_or(1);
        } else {
          // If we have no assumption to go off of, use the most pessimistic
          // range possible.
          valueUmin = 0;
          valueUmax = INT64_MAX;
          valueUdiv = 1;
        }
      }

      // The range is unsatisfiable if there is no multiple of the LCM of
      // the true divisor and the target divisor in the value's range.
      int64_t divLCM = std::lcm(udiv, valueUdiv);
      int64_t nearestUminCeil = (divLCM + valueUmin - 1) / divLCM;
      if (umin > valueUmax || umax < valueUmin || nearestUminCeil > umax) {
        neverApplies = true;
        break;
      }

      // Check if the target range is fully contained within the true range.
      // If not, the target range may sometimes not apply and we need a
      // fallback.
      if (umin > valueUmin || umax < valueUmax || valueUdiv % udiv != 0) {
        alwaysApplies = false;
      }
    }

    // Skip this range if it never applies.
    if (neverApplies) {
      LLVM_DEBUG({
        llvm::dbgs() << "Specialization range never applies:\n\t";
        llvm::dbgs() << specializationRange << "\n";
      });
      continue;
    }

    // Skip all subsequent ranges if this one always applies. Note that if this
    // range always applies, the assumptions within the IR is already a strict
    // subset of the requested ranges, meaning no change is needed to the IR
    // either.
    if (alwaysApplies) {
      LLVM_DEBUG({
        llvm::dbgs() << "Specialization range always applies:\n\t";
        llvm::dbgs() << specializationRange << "\n";
      });
      return;
    }

    // Static iteration domains should always be perfectly resolvable.
    // Assert we have at least one dynamic workload value to specialize on.
    assert(requiresSpecialization && "unexpected static iteration domain");

    builder.setInsertionPoint(currentFunction);
    StringAttr currentFunctionName = currentFunction.getSymNameAttr();

    // Use a mapping to track the newly created assumption ops.
    IRMapping mapping;
    auto clonedFunction =
        cast<func::FuncOp>(builder.clone(*currentFunction, mapping));
    StringAttr newFunctionName = symbolTable.insert(clonedFunction);

    // To avoid invalidating the workload mapping, swap the symbol names so the
    // original function body stays as |currentFunction| (the last function in
    // the fallback cascade).
    symbolTable.setSymbolName(currentFunction, newFunctionName);
    symbolTable.setSymbolName(clonedFunction, currentFunctionName);

    builder.setInsertionPointAfter(currentExport);
    auto newExport =
        cast<IREE::HAL::ExecutableExportOp>(builder.clone(*currentExport));
    int64_t newOrdinal = getAndInsertNextAvailableOrdinal(
        ordinals, currentExport.getOrdinalAttr().getInt());
    newExport.setOrdinalAttr(builder.getIndexAttr(newOrdinal));
    newExport.setSymNameAttr(newFunctionName);

    currentExport.setConditionFallback(newFunctionName);
    {
      SmallVector<Type> argTypes(
          currentExport.getWorkgroupCountBody()->getArgumentTypes());
      SmallVector<Location> locs(argTypes.size(), loc);
      Block *newCondition = builder.createBlock(
          &currentExport.getCondition(), currentExport.getCondition().begin(),
          argTypes, locs);
      builder.setInsertionPointToStart(newCondition);

      Value exportCondition =
          builder.create<arith::ConstantIntOp>(loc, 1, builder.getI1Type());

      for (auto [range, assumedSize] :
           llvm::zip(specializationRange, workloadMapping)) {
        if (!ShapedType::isDynamic(assumedSize.staticSize)) {
          continue;
        }

        // +1 for the device.
        Value workload =
            newCondition->getArgument(assumedSize.workloadOrdinal + 1);
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

        if (range.getUmin().has_value()) {
          Value uminVal = builder.create<arith::ConstantIndexOp>(
              loc, range.getUmin().value());
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ule, uminVal, workload);
          exportCondition =
              builder.create<arith::AndIOp>(loc, cmp, exportCondition);
        }
        if (range.getUmax().has_value()) {
          Value umaxVal = builder.create<arith::ConstantIndexOp>(
              loc, range.getUmax().value());
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, umaxVal, workload);
          exportCondition =
              builder.create<arith::AndIOp>(loc, cmp, exportCondition);
        }
        if (range.getUdiv().has_value()) {
          Value udivVal = builder.create<arith::ConstantIndexOp>(
              loc, range.getUdiv().value());
          Value rem = builder.create<arith::RemUIOp>(loc, workload, udivVal);
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, rem, zero);
          exportCondition =
              builder.create<arith::AndIOp>(loc, cmp, exportCondition);
        }

        if (auto originalAssumeOp = llvm::dyn_cast<IREE::Util::AssumeIntOp>(
                assumedSize.assumptionOrOrdinal.getOwner())) {
          auto clonedAssumeOp =
              cast<IREE::Util::AssumeIntOp>(mapping.lookup(originalAssumeOp));
          ArrayAttr assumptionsAttr = clonedAssumeOp.getAssumptionsAttr();
          SmallVector<Attribute> newAssumptionsLists(
              assumptionsAttr.getAsRange<Attribute>());

          // Replace the list in the assume with the one we're specializing for.
          // Single-attribute assumption lists are implicitly broadcasted to the
          // total number of callsites so this is always valid.
          newAssumptionsLists[assumedSize.assumptionOrOrdinal
                                  .getResultNumber()] =
              builder.getArrayAttr({range});
          clonedAssumeOp.setAssumptionsAttr(
              builder.getArrayAttr(newAssumptionsLists));
        } else {
          // If there is no assume already present, create a new one and replace
          // the operand of the ordinal with it. Normally all assumptions would
          // be combined into one, however this is difficult to do at this late
          // stage because it's hard to make dominance guarantees. The reason
          // for a single assume it to correlate callsites, which is irrelevant
          // here anyway so this is fine.
          OpBuilder::InsertionGuard g(builder);
          auto ordinalOp = cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(
              mapping.lookup(assumedSize.assumptionOrOrdinal.getOwner()));
          builder.setInsertionPoint(ordinalOp);
          Value assumedOperand =
              builder
                  .create<IREE::Util::AssumeIntOp>(loc, ordinalOp.getOperand(),
                                                   builder.getArrayAttr(range))
                  .getResult(0);
          ordinalOp.setOperand(assumedOperand);
        }
      }

      builder.create<IREE::HAL::ReturnOp>(loc, exportCondition);
    }
    // Current function is still the original function, just with a new symbol
    // name.
    currentExport = newExport;
  }

  return;
}

/// Walks the function |func| exported by |exportOp| and looks for a tilable
/// operation annotated with specialization ranges. The range annotation can
/// be accesses with |helper|.
static void specializeExportedFunctionByRangeAttribute(
    IREE::HAL::ExecutableExportOp exportOp, func::FuncOp func,
    SpecializationRangesAttrHelper helper,
    llvm::SmallDenseSet<int64_t> &ordinals) {
  TilingInterface specializationRoot;
  IREE::Util::MultiIntAssumptionArrayAttr specializationRanges;

  // Walk for the first op specifying specialization ranges. Its unclear what
  // to do with multiple annotations (which one takes precedence etc.) so for
  // now take the first.
  func.walk([&](TilingInterface op) {
    if (auto maybeRanges = helper.getAttr(op)) {
      specializationRanges = maybeRanges;
      specializationRoot = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!specializationRoot) {
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Specialization root:\n\t";
    llvm::dbgs() << specializationRoot << "\n";
  });

  // Remove the attribute now before it potentially gets cloned across multiple
  // function copies.
  helper.removeAttr(specializationRoot);

  specializeExportedFunction(exportOp, func, specializationRoot,
                             specializationRanges, ordinals);
}

namespace {
struct SpecializeExportsPass final
    : impl::SpecializeExportsPassBase<SpecializeExportsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variant = getOperation();

    auto *codegenDialect =
        getContext().getLoadedDialect<IREE::Codegen::IREECodegenDialect>();
    SpecializationRangesAttrHelper helper =
        codegenDialect->getSpecializationRangesAttrHelper();

    ModuleOp innerModule = variant.getInnerModule();
    if (!innerModule) {
      // Nothing to do if function definitions aren't provided.
      return;
    }

    SmallVector<IREE::HAL::ExecutableExportOp, 1> exports(
        variant.getExportOps());

    // Get the set of exported ordinals. We need to know which ordinals are
    // available when introducing new exports.
    llvm::SmallDenseSet<int64_t> ordinalSet;
    for (auto exportOp : exports) {
      IntegerAttr ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        exportOp.emitError("Missing export ordinal for specialization.");
        return signalPassFailure();
      }
      ordinalSet.insert(ordinalAttr.getInt());
    }

    for (auto exportOp : exports) {
      auto exportedFunc = llvm::dyn_cast_if_present<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(innerModule,
                                               exportOp.getSymNameAttr()));
      if (!exportedFunc || exportedFunc.isExternal()) {
        // Skip external functions.
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Specializing export:\n\t";
        llvm::dbgs() << exportOp << "\n";
      });
      specializeExportedFunctionByRangeAttribute(exportOp, exportedFunc, helper,
                                                 ordinalSet);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
