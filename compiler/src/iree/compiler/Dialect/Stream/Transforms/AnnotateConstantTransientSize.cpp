// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATECONSTANTTRANSIENTSIZEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-annotate-constant-transient-size
//===----------------------------------------------------------------------===//

// Merges a new named attribute into an existing reflection dictionary.
static DictionaryAttr
getReflectionAttrWithNamedAttr(DictionaryAttr existingAttr,
                               NamedAttribute namedAttr) {
  SmallVector<NamedAttribute> reflectionAttrs;
  if (existingAttr) {
    llvm::append_range(reflectionAttrs, existingAttr.getValue());
  }
  reflectionAttrs.push_back(namedAttr);
  return DictionaryAttr::get(namedAttr.getName().getContext(), reflectionAttrs);
}

// Analyzes a size query function to detect if it returns a constant value.
// Returns the constant IntegerAttr if detected, otherwise std::nullopt.
// Uses pattern matching to support all sources of constant values.
static std::optional<IntegerAttr>
detectConstantReturn(IREE::Util::FuncOp funcOp) {
  // Must have exactly one block (no control flow).
  if (!funcOp.getBody().hasOneBlock()) {
    return std::nullopt;
  }

  // Find the return operation.
  Block &block = funcOp.getBody().front();
  auto returnOp = dyn_cast<IREE::Util::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) {
    return std::nullopt;
  }

  // Use pattern matching to detect constant return value.
  IntegerAttr constantValue;
  if (matchPattern(returnOp.getOperand(0), m_Constant(&constantValue))) {
    return constantValue;
  }

  return std::nullopt;
}

struct AnnotateConstantTransientSizePass
    : public IREE::Stream::impl::AnnotateConstantTransientSizePassBase<
          AnnotateConstantTransientSizePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Walk all util.func operations in the module looking for ones that
    // calculate transient sizes.
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      // Check for iree.reflection attribute.
      auto reflectionAttr =
          funcOp->getAttrOfType<DictionaryAttr>("iree.reflection");
      if (!reflectionAttr) {
        continue;
      }

      // Check if already has constant annotation (user override - skip).
      if (reflectionAttr.get("iree.abi.transients.size.constant")) {
        continue;
      }

      // Get size query function reference.
      auto sizeQueryRef =
          reflectionAttr.getAs<FlatSymbolRefAttr>("iree.abi.transients.size");
      if (!sizeQueryRef) {
        continue;
      }

      // Look up size query function.
      auto sizeQueryFunc =
          symbolTable.lookup<IREE::Util::FuncOp>(sizeQueryRef.getValue());
      if (!sizeQueryFunc) {
        funcOp.emitError("referenced transient size query function '@")
            << sizeQueryRef.getValue() << "' not found in module";
        return signalPassFailure();
      }

      // Validate function signature.
      auto funcType = sizeQueryFunc.getFunctionType();

      // Must have exactly one return value.
      if (funcType.getNumResults() == 0) {
        sizeQueryFunc.emitError(
            "transient size query function must return exactly one value");
        return signalPassFailure();
      }

      // Multi-value case - emit warning and skip.
      if (funcType.getNumResults() > 1) {
        sizeQueryFunc.emitWarning("transient size query with multiple return "
                                  "values not yet supported for constant "
                                  "annotation");
        continue;
      }

      // Must return index type.
      if (!funcType.getResult(0).isIndex()) {
        sizeQueryFunc.emitError(
            "transient size query function must return index type");
        return signalPassFailure();
      }

      // Analyze function body for constant return; ignore non-constant.
      if (auto constantValue = detectConstantReturn(sizeQueryFunc)) {
        // Add annotation to reflection metadata.
        auto constantAttr =
            NamedAttribute(StringAttr::get(moduleOp.getContext(),
                                           "iree.abi.transients.size.constant"),
                           *constantValue);
        auto newReflection =
            getReflectionAttrWithNamedAttr(reflectionAttr, constantAttr);
        funcOp->setAttr("iree.reflection", newReflection);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
