// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

using AssertOp = IREE::Codegen::AssertOp;
using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;
using RootOpAttr = IREE::Codegen::RootOpAttr;

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
  AssertOp::create(builder, loc, eq, fmtMsg, ValueRange{lhs, rhs});
}

/// Helper to build a knob variable name from a prefix.
/// e.g. ("wg_", 2) -> "wg_2".
static std::string makeVarName(StringRef prefix, unsigned idx) {
  return (prefix + Twine(idx)).str();
}

/// Helper to create an i64 IntegerAttr with a fixed value.
static IntegerAttr makeIntAttr(MLIRContext *ctx, int64_t value = 0) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), value);
}

/// Helper to create an IntKnobAttr.
static IntKnobAttr makeIntKnobAttr(MLIRContext *ctx, StringRef name) {
  return IntKnobAttr::get(ctx, StringAttr::get(ctx, name));
}

/// Get unique compatible MMA attrs for matmul and conv ops.
SmallVector<Attribute> getCompatibleMMAAttrs(linalg::LinalgOp op,
                                             IREE::GPU::TargetAttr gpuTarget,
                                             const RootOpLoopInfo &loopInfo,
                                             const ContractionLikeDims &dims) {
  if (gpuTarget.getWgp().getMma().empty()) {
    return {};
  }

  SmallVector<Attribute> mmaAttrs;
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();
  SmallVector<int64_t> bounds = loopInfo.staticLoopRanges;
  Type lhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  Type rhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(1)->get());
  Type initElemType = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());

  GPUMatmulShapeType problem{bounds[dims.m.back()], bounds[dims.n.back()],
                             bounds[dims.k.back()], lhsElemType,
                             rhsElemType,           initElemType};

  auto getIntrinsic = [](IREE::GPU::MMAAttr mma) -> GPUIntrinsicType {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  };

  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    // Check if the mma intrinsic supports the problem.
    if (failed(canTargetIntrinsic(problem, getIntrinsic(mma),
                                  targetSubgroupSize, /*canUpcastAcc*/ true,
                                  /*mustBeAligned*/ false))) {
      continue;
    }

    if (!llvm::is_contained(mmaAttrs, mma)) {
      mmaAttrs.push_back(mma);
    }
  }
  return mmaAttrs;
}

/// Get contraction-like (m,n,k) dims for a linalg op.
/// Only supports contraction and convolution today.
FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims)) {
      return failure();
    }
    return ContractionLikeDims{llvm::to_vector(contractionDims->m),
                               llvm::to_vector(contractionDims->n),
                               llvm::to_vector(contractionDims->k)};
  }
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convolutionDims) || convolutionDims->outputImage.empty() ||
        convolutionDims->outputChannel.empty() ||
        convolutionDims->inputChannel.empty()) {
      return failure();
    }
    // TODO(Amily): This mapping aligns with how VectorDistribute
    // sets the dims for convs. It may be too coarse for conv
    // semantics; revisit when plumbing through conv constraint
    // generation.
    return ContractionLikeDims{llvm::to_vector(convolutionDims->outputImage),
                               llvm::to_vector(convolutionDims->outputChannel),
                               llvm::to_vector(convolutionDims->inputChannel)};
  }
  return failure();
}

/// Returns loop info for supported root ops.
std::optional<RootOpLoopInfo> getRootOpLoopInfo(Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    return RootOpLoopInfo{linalgOp.getStaticLoopRanges(),
                          linalgOp.getNumLoops(),
                          linalgOp.getIndexingMapsArray()};
  }
  return std::nullopt;
}

/// Build the VectorDistribute knobs dict for contraction-like dims.
DictionaryAttr
buildVectorDistributeKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                               const ContractionLikeDims &dims,
                               ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Build workgroup array: M and N dims get IntKnobAttr, others get 0 : i64.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, 0));
  for (ArrayRef<unsigned> dimSet : {ArrayRef(dims.m), ArrayRef(dims.n)}) {
    for (unsigned d : dimSet) {
      workgroupEntries[d] = makeIntKnobAttr(ctx, makeVarName(kKnobWgPrefix, d));
    }
  }
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));
  // Build reduction array: K dims get IntKnobAttr, others get 0 : i64.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, 0));
  for (unsigned d : dims.k) {
    reductionEntries[d] = makeIntKnobAttr(ctx, makeVarName(kKnobRedPrefix, d));
  }
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, StringAttr::get(ctx, kKnobMmaIdxName),
                         ArrayAttr::get(ctx, compatibleMMAs)));

  // Build subgroup basis with counts and mapping.
  SmallVector<NamedAttribute> subgroupBasisEntries;
  // Only innermost M and N dims get subgroup tiling, others stay 1.
  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  // A successful call to linalg::inferContractionDims guarantees that dims.m
  // and dims.n are non-empty.
  subgroupCounts[dims.m.back()] = makeIntKnobAttr(ctx, kKnobSgMCntName);
  subgroupCounts[dims.n.back()] = makeIntKnobAttr(ctx, kKnobSgNCntName);
  subgroupBasisEntries.emplace_back(kKnobCountsKey,
                                    ArrayAttr::get(ctx, subgroupCounts));
  SmallVector<Attribute> subgroupMapping;
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  subgroupBasisEntries.emplace_back(kKnobMappingKey,
                                    ArrayAttr::get(ctx, subgroupMapping));
  knobsEntries.emplace_back(kKnobSubgroupBasisKey,
                            DictionaryAttr::get(ctx, subgroupBasisEntries));

  // Add workgroup size and subgroup size at the top level.
  SmallVector<Attribute> wgSizeKnobs = {makeIntKnobAttr(ctx, kKnobWgSizeXName),
                                        makeIntKnobAttr(ctx, kKnobWgSizeYName),
                                        makeIntKnobAttr(ctx, kKnobWgSizeZName)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            makeIntKnobAttr(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit VectorDistribute constraints for contraction-like dims (matmul/conv).
/// TODO(#23535): Complete real constraint logics here.
static LogicalResult
emitVectorDistributeConstraints(OpBuilder &builder, linalg::LinalgOp linalgOp,
                                const ContractionLikeDims &dims,
                                IREE::GPU::TargetAttr gpuTarget,
                                ArrayRef<Value> smtDimArgs) {
  Location loc = linalgOp.getLoc();

  // Problem size must be divisible by tiling size.
  for (ArrayRef<unsigned> dimSet : {ArrayRef(dims.m), ArrayRef(dims.n)}) {
    for (unsigned d : dimSet) {
      std::string name = makeVarName(kKnobWgPrefix, d);
      Value wgKnob = mkKnob(builder, loc, name);
      assertDivisible(
          builder, loc, smtDimArgs[d], wgKnob,
          (kLoopRangePrefix + Twine(d) + " must be divisible by " + name)
              .str());
    }
  }
  for (unsigned d : dims.k) {
    std::string name = makeVarName(kKnobRedPrefix, d);
    Value redKnob = mkKnob(builder, loc, name);
    assertDivisible(
        builder, loc, smtDimArgs[d], redKnob,
        (kLoopRangePrefix + Twine(d) + " must be divisible by " + name).str());
  }

  return success();
}

/// Emit constraints for a single root op under the VectorDistribute pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult
emitVectorDistributeConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr) {
  // Gate on contraction-like linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || (!linalg::isaContractionOpInterface(linalgOp) &&
                    !linalg::isaConvolutionOpInterface(linalgOp))) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  std::optional<RootOpLoopInfo> loopInfo = getRootOpLoopInfo(rootOp);
  if (!loopInfo) {
    return success();
  }

  FailureOr<ContractionLikeDims> dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  DictionaryAttr knobs =
      buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  Attribute pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);

  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                         shell.smtDimArgs);
}

LogicalResult emitLLVMGPUConstraints(Attribute attr,
                                     ArrayRef<Operation *> rootOps) {
  if (rootOps.empty()) {
    return success();
  }
  // Currently only labels one root op per set.
  Operation *tunableOp = rootOps.front();
  RootOpAttr opAttr = getRootOpInfo(tunableOp);
  if (!opAttr) {
    return success();
  }

  auto pipelineAttr = cast<IREE::GPU::PipelineAttr>(attr);

  // Only VectorDistribute has constraint generation today.
  if (pipelineAttr.getValue() !=
      IREE::GPU::LoweringPipeline::VectorDistribute) {
    return success();
  }

  return emitVectorDistributeConstraintsForOp(tunableOp, opAttr);
}

} // namespace mlir::iree_compiler
