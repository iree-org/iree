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

using mlir::transform::ForeachMatchOp;
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

static bool hasConsumedArgument(ForeachMatchOp op) {
  Value operand = op->getOperand(0);
  if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto namedSequenceOp = dyn_cast<NamedSequenceOp>(parentOp)) {
      return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
                                        kArgConsumedAttrName) != nullptr;
    }
  }
  return false;
}

// Extracts the base name from a `specName`.
// - If `specName` ends with `_<number>`, the base name is everything before
// `_`.
// - Otherwise, the full name is used.
static std::string getBaseName(StringRef specName) {
  auto pos = specName.rfind('_');
  if (pos != StringRef::npos) {
    StringRef potentialBase = specName.substr(0, pos);
    unsigned suffix;
    if (!specName.substr(pos + 1).getAsInteger(10, suffix)) {
      return potentialBase.str(); // If valid number, return base name
    }
  }
  return specName.str(); // Otherwise, return full name
}

static std::string
getUniqueSpecName(StringRef specName,
                  llvm::StringMap<unsigned> &specNameCounts) {
  std::string specBaseName = getBaseName(specName);
  unsigned specNameSeenCount = specNameCounts[specBaseName]++;

  return (specNameSeenCount == 0)
             ? specBaseName
             : llvm::formatv("{}_{}", specBaseName, specNameSeenCount).str();
}

// Collects `NamedSequenceOp`s and `ForeachMatchOp`s from a given module.
// - Moves `NamedSequenceOp`s to `namedSequenceOpsToMove` if their name is not
//   `__kernel_config` or they do not have
//   the`iree_codegen.tuning_spec_entrypoint` attribute.
// - Maps `NamedSequenceOp`s that are used as matchers or actions to their
//   corresponding `ForeachMatchOp` in `namedSequenceToForeachMatch`.
static void collectNamedSequenceAndForeachMatchOpsFromModule(
    ModuleOp module, SmallVector<NamedSequenceOp> &namedSequenceOpsToMove,
    SmallVector<ForeachMatchOp> &foreachMatchOps,
    llvm::DenseMap<NamedSequenceOp, ForeachMatchOp>
        &namedSequenceToForeachMatch) {

  for (auto namedSequenceOp : module.getBody()->getOps<NamedSequenceOp>()) {
    if (namedSequenceOp.getSymName() != kKernelConfigSpecName ||
        !namedSequenceOp->hasAttr(kTuningSpecEntrypointAttrName)) {
      namedSequenceOpsToMove.push_back(namedSequenceOp);
      continue;
    }

    // Extract the single ForeachMatchOp inside namedSequenceOp.
    ForeachMatchOp foreachMatch =
        *namedSequenceOp.getOps<ForeachMatchOp>().begin();
    foreachMatchOps.push_back(foreachMatch);

    // Process matchers.
    for (auto matcher : foreachMatch.getMatchers()) {
      if (auto matcherSymRef = mlir::dyn_cast<mlir::SymbolRefAttr>(matcher)) {
        if (auto matcherOp =
                SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
                    module, matcherSymRef)) {
          namedSequenceToForeachMatch[matcherOp] = foreachMatch;
        }
      }
    }

    // Process actions.
    for (auto action : foreachMatch.getActions()) {
      if (auto actionSymRef = mlir::dyn_cast<mlir::SymbolRefAttr>(action)) {
        if (auto actionOp =
                SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
                    module, actionSymRef)) {
          namedSequenceToForeachMatch[actionOp] = foreachMatch;
        }
      }
    }
  }
}

static NamedSequenceOp createKernelConfigOp(OpBuilder &builder, Location loc,
                                            StringRef name) {
  Type anyOpType = builder.getType<transform::AnyOpType>();
  FunctionType specType =
      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});

  return builder.create<NamedSequenceOp>(loc, name, TypeAttr::get(specType),
                                         /*sym_visibility=*/StringAttr{},
                                         /*arg_attrs=*/ArrayAttr{},
                                         /*res_attrs=*/ArrayAttr{});
}

static FailureOr<NamedSequenceOp>
emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
  OpBuilder builder(module->getContext());
  builder.setInsertionPointToEnd(module.getBody());

  const bool hasConsumedSequences = llvm::any_of(specsToLink, consumesInputOp);
  Location loc = builder.getFusedLoc(llvm::map_to_vector(
      specsToLink, [](NamedSequenceOp op) { return op->getLoc(); }));
  // This code creates a named sequence operation that conforms to the
  // requirements for tuning specifications with a default entry point.
  auto newSpec = createKernelConfigOp(builder, loc, kKernelConfigSpecName);
  newSpec.setArgAttr(
      0, hasConsumedSequences ? kArgConsumedAttrName : kArgReadOnlyAttrName,
      builder.getUnitAttr());
  newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
  // TODO: Re-enable default attribute as below once new linking lands.
  // module->setAttr(kTuningSpecDefaultEntrypointAttrName,
  // builder.getUnitAttr());

  // The `__kernel_config` named sequence op in the tuning spec will be renamed
  // (e.g., `__kernel_config_1`) after linking. As a result, its parent module
  // will no longer satisfy the constraint enforced by the default tuning spec
  // attribute. To maintain correctness, remove the default tuning spec
  // attribute from the parent modules of `specsToLink`.
  for (NamedSequenceOp spec : specsToLink) {
    if (auto parentModule = spec->getParentOfType<ModuleOp>()) {
      if (parentModule->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
        parentModule->removeAttr(kTuningSpecDefaultEntrypointAttrName);
      }
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
  Type anyOpType = builder.getType<transform::AnyOpType>();
  for (NamedSequenceOp spec : specsToLink) {
    ModuleOp parentModule = spec->getParentOfType<ModuleOp>();
    assert(parentModule);
    StringAttr parentSymbol = parentModule.getSymNameAttr();
    assert(parentSymbol);
    spec.setSymName(getUniqueSpecName(spec.getSymName(), specNameCounts));

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
  SmallVector<NamedSequenceOp> namedSequenceOpsToMove;
  SmallVector<ForeachMatchOp> foreachMatchOps;
  llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> namedSequenceToForeachMatch;

  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
    collectNamedSequenceAndForeachMatchOpsFromModule(
        innerModule, namedSequenceOpsToMove, foreachMatchOps,
        namedSequenceToForeachMatch);
  }

  // Step 2-a: Make sure the name sequence names are unique, and then move
  // collected NamedSequenceOps to the top-level module.
  Type anyOpType = builder.getType<transform::AnyOpType>();
  llvm::StringMap<unsigned> specNameCounts;
  for (NamedSequenceOp op : namedSequenceOpsToMove) {
    StringRef specName = op.getSymName();
    std::string newSpecName = getUniqueSpecName(specName, specNameCounts);
    op.setSymName(newSpecName);

    // Skip updating ForeachMatchOp if the NamedSequenceOp is not used in it
    // or its name has not changed.
    if (!namedSequenceToForeachMatch.contains(op) || newSpecName == specName) {
      op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
      continue;
    }

    ForeachMatchOp foreachMatchOp = namedSequenceToForeachMatch[op];

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
      StringRef actionName = cast<SymbolRefAttr>(actionAttr).getRootReference();
      updatedActions.push_back(
          (actionName == specName)
              ? SymbolRefAttr::get(builder.getContext(), newSpecName)
              : actionAttr);
    }

    // Apply the updated matchers and actions.
    foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
    foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));

    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
  }
  // Step 2-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
  // module.
  builder.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  auto newNamedSequence =
      createKernelConfigOp(builder, loc, kKernelConfigSpecName);

  bool hasConsumedArg = llvm::any_of(foreachMatchOps, hasConsumedArgument);

  StringRef attrName =
      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
  newNamedSequence.setArgAttr(0, attrName, builder.getUnitAttr());
  newNamedSequence->setAttr(kTuningSpecEntrypointAttrName,
                            builder.getUnitAttr());
  // Indicate that the output module is a default tuning spec after merging.
  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());

  // Step 2-C: Create a new block inside the NamedSequenceOp and merge the
  // ForeachMatchOp from each inner module into one ForachMatchOp.
  SmallVector<Type, 4> resultTypes;
  llvm::append_range(resultTypes, TypeRange{anyOpType});

  SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> matcherActionPairs;
  SmallVector<Value, 4> forwardedInputs;
  for (auto foreachMatchOp : foreachMatchOps) {
    ArrayAttr matchers = foreachMatchOp.getMatchers();
    ArrayAttr actions = foreachMatchOp.getActions();

    for (size_t i = 0, e = matchers.size(); i < e; ++i) {
      matcherActionPairs.push_back(
          {cast<SymbolRefAttr>(matchers[i]), cast<SymbolRefAttr>(actions[i])});
    }
    // Collect forwarded inputs (if any).
    for (Value input : foreachMatchOp.getForwardedInputs()) {
      if (!llvm::is_contained(forwardedInputs, input)) {
        forwardedInputs.push_back(input); // Avoid duplicates.
      }
    }
  }

  SmallVector<Attribute> mergedMatchers;
  SmallVector<Attribute> mergedActions;

  for (const auto &[matcher, action] : matcherActionPairs) {
    mergedMatchers.push_back(matcher);
    mergedActions.push_back(action);
  }
  Region &region = newNamedSequence.getRegion();
  Block *body = builder.createBlock(&region, region.begin(),
                                    newNamedSequence.getArgumentTypes(), loc);
  builder.setInsertionPointToStart(body);
  auto mergedForeachMatch = builder.create<ForeachMatchOp>(
      loc, resultTypes, newNamedSequence.getArgument(0), forwardedInputs,
      /* restrictRoot = */ nullptr, /* flattenResults = */ nullptr,
      builder.getArrayAttr(mergedMatchers),
      builder.getArrayAttr(mergedActions));
  builder.create<transform::YieldOp>(loc, mergedForeachMatch->getResult(0));

  // Step 3: Remove the original inner modules after merging.
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
  int numInnerModules = 0;
  int numDefaultEntrypoint = 0;

  for (auto module : module.getBody()->getOps<ModuleOp>()) {
    ++numInnerModules;
    if (module->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
      ++numDefaultEntrypoint;
    }
  }

  // If all modules have the default attribute and there are at least two
  // modules, merge and link the default tuning specs directly.
  if (numDefaultEntrypoint == numInnerModules && numDefaultEntrypoint > 1) {
    FailureOr<NamedSequenceOp> result = emitLinkedDefaultTuningSpec(module);
    // Return successfully if merging succeeds, otherwise
    // fallback to below linking pass.
    if (succeeded(result)) {
      return result;
    }
  }

  SmallVector<NamedSequenceOp> tuningSpecs;
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
