// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

#define DEBUG_TYPE "iree-codegen-link-tuning-specs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LINKTUNINGSPECSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using mlir::transform::NamedSequenceOp;

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

static LogicalResult validateTuningSpec(NamedSequenceOp op) {
  if (!op.getResultTypes().empty()) {
    op->emitWarning() << "Tuning spec expected to have no results";
    return failure();
  }

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
    op->emitWarning() << "Tuning spec expected to have one argument of type "
                         "'!transform.any_op'";
    return failure();
  }

  if (!op.getArgAttr(0, transform::TransformDialect::kArgReadOnlyAttrName)) {
    op->emitWarning() << "Tuning spec expected to have one readonly argument";
    return failure();
  }

  return success();
}

static NamedSequenceOp
emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
  OpBuilder builder(module->getContext());
  builder.setInsertionPointToEnd(module.getBody());

  Location loc = builder.getFusedLoc(llvm::map_to_vector(
      specsToLink, [](NamedSequenceOp op) { return op->getLoc(); }));
  FunctionType specType = builder.getFunctionType(
      TypeRange{builder.getType<transform::AnyOpType>()}, TypeRange{});
  auto newSpec = builder.create<NamedSequenceOp>(
      loc, kKernelConfigSpecName, TypeAttr::get(specType),
      /*sym_visibility=*/StringAttr{},
      /*arg_attrs=*/ArrayAttr{},
      /*res_attrs*/ ArrayAttr{});
  newSpec.setArgAttr(0, transform::TransformDialect::kArgReadOnlyAttrName,
                     builder.getUnitAttr());
  newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());

  Region &region = newSpec.getRegion();
  Block *body = builder.createBlock(&region, region.begin(),
                                    newSpec.getArgumentTypes(), loc);
  builder.setInsertionPointToStart(body);

  // Emit one `transform.include` op per child tuning spec. In the future,
  // we may want to switch to a custom transform op for this to perform
  // 'short-circuring' and apply at most one tuning spec.
  Value operand = body->getArgument(0);
  for (NamedSequenceOp spec : specsToLink) {
    ModuleOp parentModule = spec->getParentOfType<ModuleOp>();
    assert(parentModule);
    StringAttr parentSymbol = parentModule.getSymNameAttr();
    assert(parentSymbol);
    auto symbol = SymbolRefAttr::get(
        parentSymbol, FlatSymbolRefAttr::get(spec.getSymNameAttr()));

    // Surpress silenceable errors so that failures to match in child tuning
    // specs can be ignored.
    builder.create<transform::IncludeOp>(
        loc, TypeRange{}, symbol, transform::FailurePropagationMode::Suppress,
        operand);
  }

  builder.create<transform::YieldOp>(loc);
  return newSpec;
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

  for (ModuleOp nested : findNestedModulesWithNamedSequences(module)) {
    llvm::append_range(tuningSpecs, findTuningSpecs(nested));
  }

  for (NamedSequenceOp spec : tuningSpecs) {
    LDBG("Found tuning spec: " << spec.getSymName());
    if (failed(validateTuningSpec(spec))) {
      return failure();
    }
  }

  if (tuningSpecs.empty()) {
    LDBG("No tuning specs found, exiting without linking");
    return NamedSequenceOp{};
  }

  return emitLinkedTuningSpec(module, tuningSpecs);
}

} // namespace mlir::iree_compiler
