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
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#define DEBUG_TYPE "iree-codegen-link-tuning-specs"

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

struct TuningSpecsToMerge {
  // Contains `NamedSequenceOp`s that either:
  //  - Are not named `__kernel_config`.
  //  - Do not have the `iree_codegen.tuning_spec_entrypoint` attribute.
  SmallVector<NamedSequenceOp> namedSequenceOpsToMove;
  // Maps a `NamedSequenceOp` to its user `ForeachMatchOp`, in which the
  // `NamedSequenceOp` is used as a matcher or action inside a
  // `ForeachMatchOp`.
  llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> namedSequenceToUser;
  // Contains any remaining SymbolOpInterface operations to import.
  SmallVector<SymbolOpInterface> otherSymbols;
};

// Populates the mapping of `NamedSequenceOp` to `ForeachMatchOp`
// by processing matchers and actions inside the given `ForeachMatchOp`.
//
// This function iterates over all matcher and action references in
// `foreachMatch`. If a reference is a valid `SymbolRefAttr` that resolves to a
// `NamedSequenceOp` within the `module`, the function records the association
// by mapping the `NamedSequenceOp` to `foreachMatch` in `namedSequenceToUser`.
static void updateForeachMatchMappings(
    ForeachMatchOp foreachMatch, ModuleOp module,
    llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> &namedSequenceToUser) {

  SmallVector<Attribute> allSymbols;
  allSymbols.append(foreachMatch.getMatchers().begin(),
                    foreachMatch.getMatchers().end());
  allSymbols.append(foreachMatch.getActions().begin(),
                    foreachMatch.getActions().end());
  for (Attribute symbolRef : allSymbols) {
    if (auto symRefAttr = dyn_cast<SymbolRefAttr>(symbolRef)) {
      if (auto namedSeqOp =
              SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
                  module, symRefAttr)) {
        namedSequenceToUser[namedSeqOp] = foreachMatch;
      }
    }
  }
}

// Collects tuning specs (`NamedSequenceOp`s) from all inner modules within the
// given input model, and organizes them into a `TuningSpecsToMerge` struct.
static TuningSpecsToMerge collectTuningSpecsToMerge(ModuleOp module) {
  TuningSpecsToMerge tuningSpecs;
  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
    for (auto namedSequenceOp :
         innerModule.getBody()->getOps<NamedSequenceOp>()) {
      if (!namedSequenceOp->hasAttr(kTuningSpecEntrypointAttrName)) {
        tuningSpecs.namedSequenceOpsToMove.push_back(namedSequenceOp);
        continue;
      }

      auto foreachMatchOps = namedSequenceOp.getOps<ForeachMatchOp>();
      ForeachMatchOp foreachMatch = getSingleElement(foreachMatchOps);
      updateForeachMatchMappings(foreachMatch, innerModule,
                                 tuningSpecs.namedSequenceToUser);
    }
    for (auto symbolOp : innerModule.getBody()->getOps<SymbolOpInterface>()) {
      if (isa<NamedSequenceOp>(symbolOp)) {
        continue;
      }
      tuningSpecs.otherSymbols.push_back(symbolOp);
    }
  }

  return tuningSpecs;
}

// Renames a `NamedSequenceOp` to resolve name conflicts caused by merging
// tuning specs.
// The name conflict resolution strategy follows below two rules:
//   1. If the `NamedSequenceOp` is inside a module with a valid symbol name,
//      its new name is prefixed with its containing module's symbol name.
//   2. If the module has no symbol name, an incrementing counter is used
//      to generate a unique prefix (e.g., `m0_`, `m1_`, etc.).
static void updateNamedSequenceOp(
    NamedSequenceOp op, OpBuilder &builder,
    llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> &namedSequenceToUser,
    llvm::DenseMap<ModuleOp, std::string> &unnamedModuleNames,
    unsigned &unnamedModuleCounter) {
  StringRef specName = op.getSymName();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  assert(parentModule);
  StringAttr parentSymbol = parentModule.getSymNameAttr();
  std::string moduleName;
  if (parentSymbol) {
    moduleName = parentSymbol.getValue().str();
  } else {
    if (unnamedModuleNames.contains(parentModule)) {
      moduleName = unnamedModuleNames[parentModule];
    } else {
      std::string newModuleName =
          llvm::formatv("m{}", unnamedModuleCounter).str();
      ++unnamedModuleCounter;
      unnamedModuleNames[parentModule] = newModuleName;
      moduleName = newModuleName;
    }
  }

  std::string newSpecName = llvm::formatv("{}_{}", moduleName, specName).str();
  op.setSymName(newSpecName);

  // Skip updating ForeachMatchOp if the NamedSequenceOp is not used in it.
  if (!namedSequenceToUser.contains(op))
    return;

  ForeachMatchOp foreachMatchOp = namedSequenceToUser[op];

  // Helper function for updating matchers or actions in associated
  // `foreachMatchOp` instances.
  auto getUpdatedSymbol = [&](Attribute attr) -> SymbolRefAttr {
    StringRef name = cast<SymbolRefAttr>(attr).getRootReference();
    return (name == specName)
               ? SymbolRefAttr::get(builder.getContext(), newSpecName)
               : cast<SymbolRefAttr>(attr);
  };

  SmallVector<Attribute> updatedMatchers;
  SmallVector<Attribute> updatedActions;
  for (Attribute matcherAttr : foreachMatchOp.getMatchers()) {
    SymbolRefAttr updatedAttr = getUpdatedSymbol(matcherAttr);
    updatedMatchers.push_back(updatedAttr);
  }

  for (Attribute actionAttr : foreachMatchOp.getActions()) {
    SymbolRefAttr updatedAttr = getUpdatedSymbol(actionAttr);
    updatedActions.push_back(updatedAttr);
  }

  // Apply the updated matchers and actions.
  foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
  foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
}

static LogicalResult resolveAndMoveNamedSequenceOps(
    ArrayRef<NamedSequenceOp> namedSequenceOpsToMove, ModuleOp module,
    OpBuilder &builder,
    llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> &namedSequenceToUser,
    ArrayRef<SymbolOpInterface> otherSymbols) {
  llvm::DenseSet<StringRef> seenNames;
  SmallVector<NamedSequenceOp> nameConflictOps;

  // Detect name conflicts across named sequence ops from differnt tuning specs.
  for (NamedSequenceOp op : namedSequenceOpsToMove) {
    StringRef name = op.getName();
    if (!seenNames.insert(name).second) {
      nameConflictOps.push_back(op);
    }
  }

  // Look for any naming conflicts among other symbols.
  for (SymbolOpInterface op : otherSymbols) {
    StringRef name = op.getName();
    if (!seenNames.insert(name).second) {
      // TODO: Support updating private symbol name conflicts. There is
      // nothing we can do for public symbols and is better done with
      // nested references.
      return failure();
    }
  }

  // Update conflicted named sequence ops.
  if (!nameConflictOps.empty()) {
    llvm::DenseMap<ModuleOp, std::string> unnamedModuleNames;
    unsigned unnamedModuleCounter = 0;
    for (NamedSequenceOp op : nameConflictOps) {
      updateNamedSequenceOp(op, builder, namedSequenceToUser,
                            unnamedModuleNames, unnamedModuleCounter);
    }
  }

  // Move all named sequence ops to the top-level module.
  for (NamedSequenceOp op : namedSequenceOpsToMove) {
    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
  }
  for (SymbolOpInterface op : otherSymbols) {
    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
  }
  return success();
}

// Retrieves the unique `ForeachMatchOp` inside the given `__kernel_config`
// named sequence op. Asserts the op has the expected `__kernel_config` name and
// `iree_codegen.tuning_spec_entrypoint` attribute.
static ForeachMatchOp
getForeachMatchOpFromKernelConfig(NamedSequenceOp namedSequenceOp) {
  assert(namedSequenceOp.getSymName() == kKernelConfigSpecName &&
         namedSequenceOp->hasAttr(kTuningSpecEntrypointAttrName) &&
         "NamedSequenceOp must have the expected symbol name `__kernel_config` "
         "and attribute `iree_codegen.tuning_spec_entrypoint`.");

  auto foreachMatchOps = namedSequenceOp.getOps<ForeachMatchOp>();
  return getSingleElement(foreachMatchOps);
}

static NamedSequenceOp createKernelConfigOp(OpBuilder &builder, Location loc,
                                            StringRef name) {
  Type anyOpType = builder.getType<transform::AnyOpType>();
  FunctionType specType =
      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});

  // Example IR generated ( with name is `__kernel_config`):
  //
  // transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
  //     -> (!transform.any_op) {
  //   transform.yield %arg0 : !transform.any_op
  // }

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

  // Step 1: Collect NamedSequenceOps from inner modules.
  TuningSpecsToMerge tuningSpecs = collectTuningSpecsToMerge(module);
  SmallVector<NamedSequenceOp> &namedSequenceOpsToMove =
      tuningSpecs.namedSequenceOpsToMove;
  llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> &namedSequenceToUser =
      tuningSpecs.namedSequenceToUser;
  SmallVector<SymbolOpInterface> &otherSymbols = tuningSpecs.otherSymbols;

  // Step 2-a: Make sure the name sequence names are unique, and then move
  // collected NamedSequenceOps to the top-level module.
  if (failed(resolveAndMoveNamedSequenceOps(namedSequenceOpsToMove, module,
                                            builder, namedSequenceToUser,
                                            otherSymbols))) {
    return failure();
  }

  // Step 2-b: Create a new entry point NamedSequenceOp called `__kernel_config`
  // in the top-level module.

  // Collect the `ForeachMatchOp`s from the entry point named sequences of
  // each inner module to merge into the new default entry point in the top
  // module.
  SmallVector<ForeachMatchOp> foreachMatchOps;
  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
    // All other NamedSequenceOps have already been moved to the top-level
    // module in Step 2-a. As a result, each inner module now contains only one
    // `__kernel_config` NamedSequenceOp that is annotated with
    // `iree_codegen.tuning_spec_entrypoint`.
    auto namedSequenceOp =
        getSingleElement(innerModule.getOps<NamedSequenceOp>());
    foreachMatchOps.push_back(
        getForeachMatchOpFromKernelConfig(namedSequenceOp));
  }

  builder.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  NamedSequenceOp newEntryPoint =
      createKernelConfigOp(builder, loc, kKernelConfigSpecName);

  bool hasConsumedArg = llvm::any_of(foreachMatchOps, hasConsumedArgument);
  StringRef attrName =
      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
  newEntryPoint.setArgAttr(0, attrName, builder.getUnitAttr());
  newEntryPoint->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
  // Indicate that the outer module is a default tuning spec after merging.
  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());

  // Step 2-c: Create a new block inside the NamedSequenceOp and merge the
  // ForeachMatchOp from each inner module into one ForachMatchOp.
  Type anyOpType = builder.getType<transform::AnyOpType>();
  SmallVector<Type, 4> resultTypes = {anyOpType};
  SmallVector<Attribute> mergedMatchers;
  SmallVector<Attribute> mergedActions;

  for (ForeachMatchOp foreachMatchOp : foreachMatchOps) {
    ArrayAttr matchers = foreachMatchOp.getMatchers();
    ArrayAttr actions = foreachMatchOp.getActions();
    for (auto [matcher, action] : llvm::zip_equal(matchers, actions)) {
      mergedMatchers.push_back(cast<SymbolRefAttr>(matcher));
      mergedActions.push_back(cast<SymbolRefAttr>(action));
    }
  }

  Region &region = newEntryPoint.getRegion();
  Block *body = builder.createBlock(&region, region.begin(),
                                    newEntryPoint.getArgumentTypes(), loc);
  builder.setInsertionPointToStart(body);
  auto mergedForeachMatch = builder.create<ForeachMatchOp>(
      loc, resultTypes, newEntryPoint.getArgument(0),
      /* forwarded_inputs = */ ValueRange(),
      /* restrictRoot = */ nullptr, /* flattenResults = */ nullptr,
      builder.getArrayAttr(mergedMatchers),
      builder.getArrayAttr(mergedActions));
  builder.create<transform::YieldOp>(loc, mergedForeachMatch->getResult(0));

  // Step 3: Remove the original inner modules after merging.
  for (auto innerModule :
       llvm::make_early_inc_range(module.getBody()->getOps<ModuleOp>())) {
    innerModule.erase();
  }

  return newEntryPoint;
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
    LDBG() << "Only " << numConsumedSpecs << " tuning specs out of "
           << tuningSpecs.size() << " total consume the input op";
    return module.emitWarning() << "Expected the argument in all tuning specs "
                                   "to be consistently readonly or consumed";
  }

  if (tuningSpecs.empty()) {
    LDBG() << "No tuning specs found, exiting without linking";
    return NamedSequenceOp{};
  }

  return emitLinkedTuningSpec(module, tuningSpecs);
}

} // namespace mlir::iree_compiler
