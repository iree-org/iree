// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/Transforms/Passes.h"

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-rocm-apply-builtin-pdl-patterns"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler::IREE::ROCM {

#define GEN_PASS_DEF_APPLYBUILTINPDLPATTERNSPASS
#include "compiler/plugins/target/ROCM/Dialect/ROCM/Transforms/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::IREE::ROCM

static LogicalResult hasAttr(PatternRewriter &rewriter, Operation *rootOp,
                             Attribute attrName) {
  auto strName = dyn_cast<StringAttr>(attrName);
  if (!strName) {
    return rewriter.notifyMatchFailure(rootOp,
                                       "expected StringAttr for attr name");
  }
  return rootOp->hasAttr(strName.strref()) ? success() : failure();
}

static LogicalResult annotateOperation(PatternRewriter &rewriter,
                                       Operation *rootOp, Attribute attrName,
                                       Attribute annotation) {
  auto strName = dyn_cast<StringAttr>(attrName);
  if (!strName) {
    return rewriter.notifyMatchFailure(rootOp,
                                       "expected StringAttr for attr name.");
  }
  rootOp->setAttr(strName.strref(), annotation);
  return success();
}

/// Helper to match contraction-like linalg ops with specific element types and
/// indexing maps.
static LogicalResult matchContraction(PatternRewriter &rewriter,
                                      Operation *rootOp, Attribute elementTypes,
                                      Attribute indexingMaps) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return rewriter.notifyMatchFailure(rootOp,
                                       "not a contraction like linalg op");
  }

  if (linalgOp.getIndexingMaps() != indexingMaps) {
    return rewriter.notifyMatchFailure(rootOp, "indexing maps mismatch");
  }

  SmallVector<Attribute> opElemTypes =
      llvm::map_to_vector(rootOp->getOperandTypes(), [](Type t) -> Attribute {
        return TypeAttr::get(getElementTypeOrSelf(t));
      });
  if (rewriter.getArrayAttr(opElemTypes) != elementTypes) {
    return rewriter.notifyMatchFailure(rootOp, "element types mismatch");
  }
  return success();
}

/// PDL utility that checks whether the size of the dimension of the provided
/// value is a multiple of the divisor.
static LogicalResult dimIsMultipleOf(PatternRewriter &rewriter, Value value,
                                     Attribute dim, Attribute divisor) {
  auto shapedType = dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType) {
    return failure();
  }
  auto dimInt = dyn_cast<IntegerAttr>(dim);
  if (!dim) {
    return failure();
  }
  auto divisorInt = dyn_cast<IntegerAttr>(divisor);
  if (!divisor) {
    return failure();
  }
  ValueBoundsConstraintSet::Variable dimVar(value, dimInt.getInt());

  // Check if value[dim] % size == 0. There are a couple of options for how
  // to do this (e.g. mul(floordiv)). Affine map canonicalizations are good
  // at dropping terms that statically divide the mod RHS so we go with this
  // one.
  MLIRContext *ctx = value.getContext();
  AffineMap modMap =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                     getAffineSymbolExpr(0, ctx) %
                         getAffineConstantExpr(divisorInt.getInt(), ctx));
  ValueBoundsConstraintSet::Variable modVar(modMap, {dimVar});
  Builder b(ctx);
  FailureOr<bool> maybeFailed = ValueBoundsConstraintSet::areEqual(
      modVar, OpFoldResult{b.getIndexAttr(0)});
  if (failed(maybeFailed) || !maybeFailed.value()) {
    return failure();
  }
  return success();
}

/// PDL utility that checks whether the size of the dimension of the provided
/// value is within the specified range.
static LogicalResult dimIsBound(PatternRewriter &rewriter, Value value,
                                Attribute dim, Attribute lowerBound,
                                Attribute upperBound) {
  auto shapedType = dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType) {
    return failure();
  }
  auto dimInt = dyn_cast<IntegerAttr>(dim);
  if (!dimInt) {
    return failure();
  }
  if (auto lowerBoundInt = dyn_cast<IntegerAttr>(lowerBound)) {
    FailureOr<int64_t> constantLb =
        ValueBoundsConstraintSet::computeConstantBound(
            presburger::BoundType::LB, {value, /*dim=*/dimInt.getInt()},
            /*stopCondition=*/nullptr, /*closedLB=*/true);
    if (failed(constantLb)) {
      return failure();
    }
    if (lowerBoundInt.getInt() > constantLb.value()) {
      return failure();
    }
  }
  if (auto upperBoundInt = dyn_cast<IntegerAttr>(upperBound)) {
    FailureOr<int64_t> constantUb =
        ValueBoundsConstraintSet::computeConstantBound(
            presburger::BoundType::UB, {value, /*dim=*/dimInt.getInt()},
            /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(constantUb)) {
      return failure();
    }
    if (upperBoundInt.getInt() < constantUb.value()) {
      return failure();
    }
  }
  return success();
}

// Populate patterns from builtin.
static LogicalResult
populatePDLModuleFromBuiltin(MLIRContext *context, RewritePatternSet &patterns,
                             llvm::StringRef builtinString) {
  // Rewrite pattern sets take a unique_ptr to the pdl module and erase it once
  // frozen pattern set is destroyed. This means we need a full copy of the PDL
  // module for every invocation of this pass.
  //
  // TODO: Cache unique frozen rewrite pattern sets somewhere to avoid repeat
  // parsing and conversion to PDL bytecode.
  PDLPatternModule pdlModule = OwningOpRef<ModuleOp>(
      parseSourceString<ModuleOp>(builtinString, context));
  if (!pdlModule.getModule()) {
    // Fail on parser error.
    return failure();
  }
  // Constraints are registered to the PDL module, and linking the modules
  // together is likely slower than just registering a few redundant functions.
  pdlModule.registerConstraintFunction("hasAttr", hasAttr);
  pdlModule.registerConstraintFunction("dimIsBound", dimIsBound);
  pdlModule.registerConstraintFunction("dimIsMultipleOf", dimIsMultipleOf);
  pdlModule.registerConstraintFunction("matchContraction", matchContraction);
  pdlModule.registerRewriteFunction("annotateOperation", annotateOperation);
  patterns.insert(std::move(pdlModule));
  return success();
}

namespace {

class ApplyBuiltinPDLPatternsPass
    : public iree_compiler::IREE::ROCM::impl::ApplyBuiltinPDLPatternsPassBase<
          ApplyBuiltinPDLPatternsPass> {
public:
  using iree_compiler::IREE::ROCM::impl::ApplyBuiltinPDLPatternsPassBase<
      ApplyBuiltinPDLPatternsPass>::ApplyBuiltinPDLPatternsPassBase;

  LogicalResult initialize(MLIRContext *context) override {
    // Populate the specialization patterns from the list of targets.
    auto rocmDialect = context->getOrLoadDialect<IREE::ROCM::ROCMDialect>();
    RewritePatternSet tmpPatterns(context);
    if (enableSpecialization) {
      for (std::string target : targets) {
        std::string builtinName =
            llvm::formatv("specialization_patterns_{}.mlir", target);
        std::optional<StringRef> maybeBuiltin =
            rocmDialect->getBuiltin(builtinName);
        if (!maybeBuiltin) {
          // Skip when no patterns are present.
          continue;
        }
        if (failed(populatePDLModuleFromBuiltin(context, tmpPatterns,
                                                maybeBuiltin.value()))) {
          return failure();
        }
      }
    }
    if (enableTensorUKernels) {
      for (std::string target : targets) {
        std::string builtinName =
            llvm::formatv("ukernel_patterns_{}.mlir", target);
        std::optional<StringRef> maybeBuiltin =
            rocmDialect->getBuiltin(builtinName);
        if (!maybeBuiltin) {
          // Skip when no patterns are present.
          continue;
        }
        if (failed(populatePDLModuleFromBuiltin(context, tmpPatterns,
                                                maybeBuiltin.value()))) {
          return failure();
        }
      }
    }
    patterns = std::move(tmpPatterns);
    return success();
  }

  void runOnOperation() override {
    // If there is nothing to do then return.
    if (!patterns.getPDLByteCode()) {
      return;
    }

    // Apply the patterns.
    auto operation = getOperation();
    if (failed(applyPatternsGreedily(operation, patterns))) {
      operation->emitOpError("failed to apply builtin specialization patterns");
      return signalPassFailure();
    }
  }

private:
  /// Loaded PDL patterns
  FrozenRewritePatternSet patterns;
};

} // namespace
