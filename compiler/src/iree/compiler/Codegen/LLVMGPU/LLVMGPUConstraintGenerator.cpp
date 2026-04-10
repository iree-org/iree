// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

// TODO(#23535): These constraints are VERY incomplete -- they only emit
// workgroup tile divisibility. Full VectorDistribute constraints (MMA
// alignment, subgroup counts, shared memory, load distribution, etc.)
// will be added in follow-up patches.

/// Assert: lhs % rhs == 0, with format args for diagnostics.
static void assertDivisible(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs, StringRef msg) {
  Value zero = mkIntConst(builder, loc, 0);
  Value rem = smt::IntModOp::create(builder, loc, lhs, rhs);
  Value eq = smt::EqOp::create(builder, loc, rem, zero);
  std::string fmtMsg = (msg + " ({} % {} == 0)").str();
  IREE::Codegen::AssertOp::create(builder, loc, eq, fmtMsg,
                                  ValueRange{lhs, rhs});
}

using IntKnobAttr = IREE::Codegen::IntKnobAttr;

/// Build a flat knobs dict with workgroup tile knobs for each loop dim.
static DictionaryAttr buildKnobsDict(MLIRContext *ctx, unsigned numLoops) {
  SmallVector<Attribute> workgroupEntries;
  for (unsigned d = 0; d < numLoops; ++d) {
    workgroupEntries.push_back(IntKnobAttr::get(ctx, ("wg_" + Twine(d)).str()));
  }
  SmallVector<NamedAttribute> knobsEntries;
  knobsEntries.emplace_back("workgroup", ArrayAttr::get(ctx, workgroupEntries));
  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit divisibility constraints: dim % tile == 0 for each loop dimension.
static LogicalResult emitConstraints(OpBuilder &builder, Operation *rootOp,
                                     ArrayRef<Value> smtDimArgs) {
  Location loc = rootOp->getLoc();
  unsigned numLoops = smtDimArgs.size();

  SmallVector<Value> wgKnobs;
  for (unsigned d = 0; d < numLoops; ++d) {
    wgKnobs.push_back(mkKnob(builder, loc, ("wg_" + Twine(d)).str()));
  }

  for (unsigned d = 0; d < numLoops; ++d) {
    assertDivisible(
        builder, loc, smtDimArgs[d], wgKnobs[d],
        llvm::formatv("dim_{} must be divisible by wg_{}", d, d).str());
  }

  return success();
}

static LogicalResult
emitContractionConstraints(OpBuilder &builder, Operation *rootOp,
                           const ConstraintsOpShell &shell) {
  return emitConstraints(builder, rootOp, shell.smtDimArgs);
}

static LogicalResult
emitConvolutionConstraints(OpBuilder &builder, Operation *rootOp,
                           const ConstraintsOpShell &shell) {
  return emitConstraints(builder, rootOp, shell.smtDimArgs);
}

static LogicalResult emitAttentionConstraints(OpBuilder &builder,
                                              Operation *rootOp,
                                              const ConstraintsOpShell &shell) {
  return emitConstraints(builder, rootOp, shell.smtDimArgs);
}

/// Emit constraints for a single root op under VectorDistribute pipeline.
/// Supported root ops are AttentionOp, linalg contraction ops, and linalg
/// convolution ops.
static LogicalResult emitConstraintsForOp(Operation *rootOp,
                                          Attribute pipelineAttr) {
  unsigned numLoops = 0;
  SmallVector<AffineMap> indexingMaps;

  if (isa<IREE::LinalgExt::AttentionOp>(rootOp)) {
    auto fusionOp = cast<IREE::LinalgExt::LinalgFusionOpInterface>(rootOp);
    numLoops = fusionOp.getNumLoops();
    indexingMaps = fusionOp.getIndexingMapsArray();
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    numLoops = linalgOp.getNumLoops();
    indexingMaps = linalgOp.getIndexingMapsArray();
  } else {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs = buildKnobsDict(ctx, numLoops);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, getRootOpInfo(rootOp),
                               pipelineAttr, knobs, numLoops, indexingMaps);

  if (isa<IREE::LinalgExt::AttentionOp>(rootOp)) {
    return emitAttentionConstraints(builder, rootOp, shell);
  }

  auto linalgOp = cast<linalg::LinalgOp>(rootOp);
  if (linalg::isaContractionOpInterface(linalgOp)) {
    return emitContractionConstraints(builder, rootOp, shell);
  }
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    return emitConvolutionConstraints(builder, rootOp, shell);
  }

  return success();
}

LogicalResult emitLLVMGPUConstraints(Attribute attr,
                                     ArrayRef<Operation *> rootOps) {
  auto pipelineAttr = cast<IREE::GPU::PipelineAttr>(attr);

  // Only VectorDistribute has constraint generation today.
  if (pipelineAttr.getValue() !=
      IREE::GPU::LoweringPipeline::VectorDistribute) {
    return success();
  }

  // KernelConfig.cpp currently only labels one root op per set.
  Operation *tunableOp = rootOps.front();
  if (!tunableOp) {
    return success();
  }

  return emitConstraintsForOp(tunableOp, pipelineAttr);
}

} // namespace mlir::iree_compiler
