// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"

#define DEBUG_TYPE "iree-codegen-link-tuning-specs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LINKTUNINGSPECSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using mlir::transform::NamedSequenceOp;
constexpr StringLiteral kArgConsumedAttrName =
    mlir::transform::TransformDialect::kArgConsumedAttrName;
constexpr StringLiteral kArgReadOnlyAttrName =
    mlir::transform::TransformDialect::kArgReadOnlyAttrName;

static SmallVector<ModuleOp>
findNestedModulesWithNamedSequences(ModuleOp module) {
  Block *body = module.getBody();
  return llvm::filter_to_vector(body->getOps<ModuleOp>(), [](ModuleOp op) {
    return op.getSymName().has_value() &&
           op->hasAttr(transform::TransformDialect::kWithNamedSequenceAttrName);
  });
}

static SmallVector<NamedSequenceOp> findTuningSpecs(ModuleOp module) {
  Block *body = module.getBody();
  return llvm::filter_to_vector(
      body->getOps<NamedSequenceOp>(), [](NamedSequenceOp op) {
        return op->hasAttr(kTuningSpecEntrypointAttrName);
      });
}

static bool consumesInputOp(NamedSequenceOp op) {
  if (op.getArgAttr(0, kArgConsumedAttrName)) {
    return true;
  }
  return false;
}

static FailureOr<NamedSequenceOp>
emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
  OpBuilder builder(module->getContext());
  builder.setInsertionPointToEnd(module.getBody());

  const bool hasConsumedSequences = llvm::any_of(specsToLink, consumesInputOp);
  Location loc = builder.getFusedLoc(llvm::map_to_vector(
      specsToLink, [](NamedSequenceOp op) { return op->getLoc(); }));
  Type anyOpType = builder.getType<transform::AnyOpType>();
  FunctionType specType =
      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});
  // This code creates a named sequence operation that conforms to the
  // requirements for tuning specifications with a default entry point.
  auto newSpec = builder.create<NamedSequenceOp>(
      loc, kKernelConfigSpecName, TypeAttr::get(specType),
      /*sym_visibility=*/StringAttr{},
      /*arg_attrs=*/ArrayAttr{},
      /*res_attrs*/ ArrayAttr{});
  newSpec.setArgAttr(
      0, hasConsumedSequences ? kArgConsumedAttrName : kArgReadOnlyAttrName,
      builder.getUnitAttr());
  newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
  // TODO: Re-enable default attribute as below once new linking lands.
  // module->setAttr(kTuningSpecDefaultEntrypointAttrName,
  // builder.getUnitAttr());

  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
    // Remove the default tuning spec attribute from inner modules,
    // as the top-level module is attached with default attribute.
    if (innerModule->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
      innerModule->removeAttr(kTuningSpecDefaultEntrypointAttrName);
    }
  }

  Region &region = newSpec.getRegion();
  Block *body = builder.createBlock(&region, region.begin(),
                                    newSpec.getArgumentTypes(), loc);
  builder.setInsertionPointToStart(body);

  // Make sure spec names are unique to work around a transform dialect
  // interpreter bug (`transform.include` does not handle name collisions
  // correctly): https://github.com/llvm/llvm-project/issues/119578.
  llvm::StringMap<unsigned> specNameCounts;
  // Reserve the name for the outermost entrypoint.
  specNameCounts[kKernelConfigSpecName] = 1;

  // Emit one `transform.include` op per child tuning spec. In the future,
  // we may want to switch to a custom transform op for this to perform
  // 'short-circuring' and apply at most one tuning spec.
  Value operand = body->getArgument(0);
  for (NamedSequenceOp spec : specsToLink) {
    ModuleOp parentModule = spec->getParentOfType<ModuleOp>();
    assert(parentModule);
    StringAttr parentSymbol = parentModule.getSymNameAttr();
    assert(parentSymbol);
    StringRef specName = spec.getSymName();
    unsigned specNameSeenCount = specNameCounts[specName]++;
    if (specNameSeenCount > 0) {
      spec.setSymName(
          llvm::formatv("{}_{}", specName, specNameSeenCount).str());
    }

    auto symbol = SymbolRefAttr::get(
        parentSymbol, FlatSymbolRefAttr::get(spec.getSymNameAttr()));

    // Surpress silenceable errors so that failures to match in child tuning
    // specs can be ignored.
    operand = builder
                  .create<transform::IncludeOp>(
                      loc, anyOpType, symbol,
                      transform::FailurePropagationMode::Suppress, operand,
                      /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr)
                  .getResults()
                  .front();
  }

  builder.create<transform::YieldOp>(loc, operand);

  if (failed(mlir::verify(module))) {
    return module.emitError("Linked tuning spec failed to verify");
  }

  return newSpec;
}

static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
  OpBuilder builder(module.getContext());
  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
  // (either as a matcher or an action). It ensures
  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
  // ForeachMatchOp is also updated.
  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
      foreachMatchMap;

  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
    for (auto namedSequenceOp :
         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
        transform::ForeachMatchOp foreachMatch = nullptr;
        int matchCount = 0;
        // Iterate directly over ForeachMatchOp within kernelConfig.
        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
          if (!foreachMatch) {
            foreachMatch = op;
          }
          matchCount++;
        }

        // Return failure if multiple occurrences exist.
        if (matchCount > 1) {
          return failure();
        }
        // Return failure if not foreach match op found.
        if (!foreachMatch)
          return failure();

        foreachMatchOps.push_back(foreachMatch);

        for (auto matcher : foreachMatch.getMatchers()) {
          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
                    SymbolTable::lookupNearestSymbolFrom(innerModule,
                                                         matcherSymRef))) {
              if (!foreachMatchMap.count(matcherOp)) {
                foreachMatchMap[matcherOp] = foreachMatch;
              }
            }
          }
        }
        for (auto action : foreachMatch.getActions()) {
          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
                    SymbolTable::lookupNearestSymbolFrom(innerModule,
                                                         actionSymRef))) {
              if (!foreachMatchMap.count(actionOp)) {
                foreachMatchMap[actionOp] = foreachMatch;
              }
            }
          }
        }
      } else {
        namedSequenceOpsToMove.push_back(namedSequenceOp);
      }
    }
  }

  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
  // merging.
  SmallVector<Type, 4> expectedResultTypes =
      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());

  for (auto foreachMatchOp : foreachMatchOps) {
    SmallVector<Type, 4> currentResultTypes =
        llvm::to_vector<4>(foreachMatchOp.getResultTypes());

    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
      return failure();
    }
  }

  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
  // `flattenResults` attributes.
  UnitAttr restrictRoot = nullptr;
  UnitAttr flattenResults = nullptr;
  bool hasMismatchAttr = false;

  for (auto foreachMatchOp : foreachMatchOps) {
    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();

    if (!restrictRoot) {
      restrictRoot = currentRestrictRoot; // First encountered value.
    } else if (restrictRoot != currentRestrictRoot) {
      hasMismatchAttr = true;
      break; // Exit early when a mismatch is found.
    }

    if (!flattenResults) {
      flattenResults = currentFlattenResults; // First encountered value.
    } else if (flattenResults != currentFlattenResults) {
      hasMismatchAttr = true;
      break; // Exit early when a mismatch is found.
    }
  }

  // If there's a mismatch in attributes, do not merge.
  if (hasMismatchAttr) {
    return failure();
  }

  llvm::StringMap<unsigned> specNameCounts;
  // Step 3-a: Make sure the name sequence names are unique, and then move
  // collected NamedSequenceOps to the top-level module.
  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
    StringRef specName = op.getSymName();
    unsigned specNameSeenCount = specNameCounts[specName]++;
    std::string newSpecName = specName.str();
    if (specNameSeenCount > 0) {
      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
      op.setSymName(newSpecName);
    }

    // Only update ForeachMatchOp if there's a reference and the name has
    // changed.
    if (foreachMatchMap.count(op) && newSpecName != specName) {
      transform::ForeachMatchOp foreachMatchOp = foreachMatchMap[op];

      SmallVector<Attribute> updatedMatchers, updatedActions;
      for (auto matcherAttr : foreachMatchOp.getMatchers()) {
        StringRef matcherName =
            cast<SymbolRefAttr>(matcherAttr).getRootReference();
        updatedMatchers.push_back(
            (matcherName == specName)
                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
                : matcherAttr);
      }

      for (auto actionAttr : foreachMatchOp.getActions()) {
        StringRef actionName =
            cast<SymbolRefAttr>(actionAttr).getRootReference();
        updatedActions.push_back(
            (actionName == specName)
                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
                : actionAttr);
      }

      // Apply the updated matchers and actions.
      foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
      foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
    }
    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
  }

  // Step 3-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
  // module.
  builder.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  Type anyOpType = builder.getType<transform::AnyOpType>();
  FunctionType seqType =
      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});

  auto newNamedSequence = builder.create<transform::NamedSequenceOp>(
      loc, kKernelConfigSpecName, TypeAttr::get(seqType),
      /*sym_visibility=*/StringAttr{},
      /*arg_attrs=*/ArrayAttr{},
      /*res_attrs*/ ArrayAttr{});

  bool hasConsumedArg =
      llvm::any_of(foreachMatchOps, [](transform::ForeachMatchOp op) {
        Value operand = op->getOperand(0);
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
          Operation *parentOp = blockArg.getOwner()->getParentOp();
          if (auto namedSequenceOp =
                  mlir::dyn_cast<transform::NamedSequenceOp>(parentOp)) {
            return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
                                              kArgConsumedAttrName) != nullptr;
          }
        }
        return false;
      });

  StringRef attrName =
      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
  newNamedSequence.setArgAttr(0, attrName, builder.getUnitAttr());
  newNamedSequence->setAttr(kTuningSpecEntrypointAttrName,
                            builder.getUnitAttr());
  // Indicate the output module is a default tuning spec after merging.
  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());

  // Step 3-C: Create a new block inside the NamedSequenceOp and merging
  // ForeachMatchOp from each inner modules into one ForachMatchOp.
  SmallVector<Type, 4> resultTypes;
  llvm::append_range(resultTypes, expectedResultTypes);

  SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> matcherActionPairs;
  SmallVector<Value, 4> forwardedInputs;
  for (auto foreachMatchOp : foreachMatchOps) {
    ArrayAttr matchers = foreachMatchOp.getMatchers();
    ArrayAttr actions = foreachMatchOp.getActions();

    for (size_t i = 0; i < matchers.size(); i++) {
      matcherActionPairs.push_back({mlir::cast<SymbolRefAttr>(matchers[i]),
                                    mlir::cast<SymbolRefAttr>(actions[i])});
    }
    // Collect forwarded inputs (if any).
    for (Value input : foreachMatchOp.getForwardedInputs()) {
      if (llvm::find(forwardedInputs, input) == forwardedInputs.end()) {
        forwardedInputs.push_back(input); // Avoid duplicates
      }
    }
  }

  SmallVector<Attribute> mergedMatchers;
  SmallVector<Attribute> mergedActions;

  for (const auto &pair : matcherActionPairs) {
    mergedMatchers.push_back(pair.first);
    mergedActions.push_back(pair.second);
  }
  Region &region = newNamedSequence.getRegion();
  Block *body = builder.createBlock(&region, region.begin(),
                                    newNamedSequence.getArgumentTypes(), loc);
  builder.setInsertionPointToStart(body);
  auto mergedForeachMatch = builder.create<transform::ForeachMatchOp>(
      loc, resultTypes, newNamedSequence.getArgument(0), forwardedInputs,
      restrictRoot, flattenResults, builder.getArrayAttr(mergedMatchers),
      builder.getArrayAttr(mergedActions));
  builder.create<transform::YieldOp>(loc, mergedForeachMatch->getResult(0));

  // Step 4: Remove the original inner modules after merging.
  for (auto innerModule :
       llvm::make_early_inc_range(module.getBody()->getOps<ModuleOp>())) {
    innerModule.erase();
  }

  return newNamedSequence;
}

struct LinkTuningSpecsPass final
    : impl::LinkTuningSpecsPassBase<LinkTuningSpecsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    if (failed(linkTuningSpecs(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
  SmallVector<NamedSequenceOp> tuningSpecs;

  int matchingModules = 0;
  int totalModules = 0;

  for (auto module : module.getBody()->getOps<ModuleOp>()) {
    totalModules++;
    if (module->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
      matchingModules++;
    }
  }

  // If all modules have the default attribute and there are at least two
  // modules, merge and link the default tuning specs directly.
  if (matchingModules == totalModules && matchingModules > 1) {
    FailureOr<NamedSequenceOp> result = emitLinkedDefaultTuningSpec(module);
    // Return successfully if merging succeeds, otherwise
    // fallback to below linking pass.
    if (succeeded(result)) {
      return result;
    }
  }

  for (ModuleOp nested : findNestedModulesWithNamedSequences(module)) {
    llvm::append_range(tuningSpecs, findTuningSpecs(nested));
  }

  size_t numConsumedSpecs = llvm::count_if(tuningSpecs, consumesInputOp);
  if (numConsumedSpecs > 0 && numConsumedSpecs != tuningSpecs.size()) {
    LDBG("Only " << numConsumedSpecs << " tuning specs out of "
                 << tuningSpecs.size() << " total consume the input op");
    return module.emitWarning() << "Expected the argument in all tuning specs "
                                   "to be consistently readonly or consumed";
  }

  if (tuningSpecs.empty()) {
    LDBG("No tuning specs found, exiting without linking");
    return NamedSequenceOp{};
  }

  return emitLinkedTuningSpec(module, tuningSpecs);
}

} // namespace mlir::iree_compiler
