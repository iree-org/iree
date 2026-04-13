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
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
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
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;

/// Helper to create an i64 IntegerAttr with a fixed value.
static Attribute makeIntAttr(MLIRContext *ctx, int64_t value = 0) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), value);
}

/// Helper to create a IntKnobAttr.
static Attribute makeIntKnob(MLIRContext *ctx, StringRef name) {
  return IREE::Codegen::IntKnobAttr::get(ctx, StringAttr::get(ctx, name));
}

/// Get compatible MMA attrs for the given target and element types.
/// Uses the same filtering as KernelConfig.cpp: subgroup size and
/// distribution mapping kind, plus element type compatibility.
/// Returns unique Attribute objects (MMAAttr and VirtualMMAAttr).
static SmallVector<Attribute>
getCompatibleMMAAttrs(IREE::GPU::TargetAttr gpuTarget, Type lhsElemType,
                      Type rhsElemType, Type accElemType, MLIRContext *ctx,
                      bool includeVirtual = false) {
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();

  SmallVector<Attribute> attrs;
  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    // Filter 1: Subgroup size.
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // Filter 2: Distribution mapping.
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    // Filter 3: Element type compatibility.
    auto [aType, bType, cType] = mma.getABCElementTypes();
    if (aType != lhsElemType || bType != rhsElemType) {
      continue;
    }
    // Allow accumulator upcasting: MMA acc type can be narrower.
    if (cType != accElemType &&
        cType.getIntOrFloatBitWidth() > accElemType.getIntOrFloatBitWidth()) {
      continue;
    }

    Attribute mmaAttr = Attribute(mma);
    if (!llvm::is_contained(attrs, mmaAttr)) {
      attrs.push_back(mmaAttr);
    }

    if (!includeVirtual) {
      continue;
    }
    for (IREE::GPU::VirtualMMAIntrinsic vmma : mma.getVirtualIntrinsics()) {
      auto vmmaAttr = Attribute(IREE::GPU::VirtualMMAAttr::get(ctx, vmma));
      if (!llvm::is_contained(attrs, vmmaAttr)) {
        attrs.push_back(vmmaAttr);
      }
    }
  }
  return attrs;
}

/// Get compatible MMA attrs for a linalg op.
static SmallVector<Attribute>
getCompatibleMMAAttrsForLinalgOp(IREE::GPU::TargetAttr gpuTarget,
                                 linalg::LinalgOp linalgOp) {
  Type lhsElemType =
      getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get().getType());
  Type rhsElemType =
      getElementTypeOrSelf(linalgOp.getDpsInputOperand(1)->get().getType());
  Type accElemType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  return getCompatibleMMAAttrs(gpuTarget, lhsElemType, rhsElemType, accElemType,
                               linalgOp.getContext());
}

/// Contraction-like dimension classification used by both matmul and conv.
struct ContractionLikeDims {
  SmallVector<unsigned> m;
  SmallVector<unsigned> n;
  SmallVector<unsigned> k;
};

/// Get contraction-like (m,n,k) dims for a linalg op.
/// Only supports contraction and convolution today.
static FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    auto contractionDims = linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims)) {
      return failure();
    }
    return ContractionLikeDims{
        {contractionDims->m.begin(), contractionDims->m.end()},
        {contractionDims->n.begin(), contractionDims->n.end()},
        {contractionDims->k.begin(), contractionDims->k.end()}};
  }
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    auto convolutionDims = linalg::inferConvolutionDims(linalgOp);
    if (failed(convolutionDims) || convolutionDims->outputImage.empty() ||
        convolutionDims->outputChannel.empty() ||
        convolutionDims->inputChannel.empty()) {
      return failure();
    }
    // Maps outputImage→M, outputChannel→N, inputChannel→K.
    return ContractionLikeDims{{convolutionDims->outputImage.begin(),
                                convolutionDims->outputImage.end()},
                               {convolutionDims->outputChannel.begin(),
                                convolutionDims->outputChannel.end()},
                               {convolutionDims->inputChannel.begin(),
                                convolutionDims->inputChannel.end()}};
  }
  return failure();
}

/// Problem size, loop count, and indexing maps for a root op.
struct RootOpLoopInfo {
  SmallVector<int64_t> staticLoopRanges;
  unsigned numLoops;
  SmallVector<AffineMap> indexingMaps;
};

/// Returns loop info for supported root ops.
static std::optional<RootOpLoopInfo> getRootOpLoopInfo(Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    return RootOpLoopInfo{linalgOp.getStaticLoopRanges(),
                          linalgOp.getNumLoops(),
                          linalgOp.getIndexingMapsArray()};
  }
  return std::nullopt;
}

/// Build the VectorDistribute knobs dict for contraction-like dims.
static DictionaryAttr
buildVectorDistributeKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                               const ContractionLikeDims &dims,
                               ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Build workgroup array: M and N dims get IntKnobAttr, others get 0 : i64.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, 0));
  for (ArrayRef<unsigned> dimSet : {ArrayRef(dims.m), ArrayRef(dims.n)}) {
    for (unsigned d : dimSet) {
      workgroupEntries[d] = makeIntKnob(ctx, "wg_" + std::to_string(d));
    }
  }
  knobsEntries.emplace_back("workgroup", ArrayAttr::get(ctx, workgroupEntries));

  // Build reduction array: K dims get IntKnobAttr, others get 0 : i64.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, 0));
  for (unsigned d : dims.k) {
    reductionEntries[d] = IntKnobAttr::get(ctx, "red_" + std::to_string(d));
  }
  knobsEntries.emplace_back("reduction", ArrayAttr::get(ctx, reductionEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      "mma_kind", OneOfKnobAttr::get(ctx, StringAttr::get(ctx, "mma_idx"),
                                     ArrayAttr::get(ctx, compatibleMMAs)));

  // Build subgroup basis with counts and mapping.
  SmallVector<NamedAttribute> subgroupBasisEntries;
  // Only innermost M and N dims get subgroup tiling, others stay 1.
  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  // A successful call to linalg::inferContractionDims guarantees that dims.m
  // and dims.n are non-empty.
  subgroupCounts[dims.m.back()] = makeIntKnob(ctx, "sg_m_cnt");
  subgroupCounts[dims.n.back()] = makeIntKnob(ctx, "sg_n_cnt");
  subgroupBasisEntries.emplace_back("counts",
                                    ArrayAttr::get(ctx, subgroupCounts));
  SmallVector<Attribute> subgroupMapping;
  for (size_t i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  subgroupBasisEntries.emplace_back("mapping",
                                    ArrayAttr::get(ctx, subgroupMapping));
  knobsEntries.emplace_back("subgroup_basis",
                            DictionaryAttr::get(ctx, subgroupBasisEntries));

  // Add workgroup size and subgroup size at the top level.
  SmallVector<Attribute> wgSizeKnobs;
  wgSizeKnobs.push_back(makeIntKnob(ctx, "wg_x"));
  wgSizeKnobs.push_back(makeIntKnob(ctx, "wg_y"));
  wgSizeKnobs.push_back(makeIntKnob(ctx, "wg_z"));
  knobsEntries.emplace_back("workgroup_size", ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back("subgroup_size", makeIntKnob(ctx, "sg_size"));

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

/// Emit VectorDistribute constraints for contraction-like dims (matmul/conv).
/// TODO(#23535): Complete real constraint logics here.
static LogicalResult
emitVectorDistributeConstraints(OpBuilder &builder, linalg::LinalgOp linalgOp,
                                const ContractionLikeDims &dims,
                                IREE::GPU::TargetAttr gpuTarget,
                                ArrayRef<Value> smtDimArgs) {
  Location loc = linalgOp.getLoc();
  DenseMap<unsigned, Value> wgKnobs;
  for (unsigned d : dims.m) {
    wgKnobs[d] = mkKnob(builder, loc, "wg_" + std::to_string(d));
  }
  for (unsigned d : dims.m) {
    assertDivisible(
        builder, loc, smtDimArgs[d], wgKnobs[d],
        ("dim_" + Twine(d) + " must be divisible by wg_" + Twine(d)).str());
  }

  return success();
}

/// Emit constraints for a single root op under the VectorDistribute pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult
emitVectorDistributeConstraintsForOp(Operation *rootOp,
                                     IREE::Codegen::RootOpAttr rootOpAttr) {
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

  auto dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  auto compatibleMMAs = getCompatibleMMAAttrsForLinalgOp(gpuTarget, linalgOp);
  DictionaryAttr knobs =
      buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  Attribute pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);

  auto shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                         shell.smtDimArgs);
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
  auto opAttr = tunableOp->getAttrOfType<IREE::Codegen::RootOpAttr>("root_op");
  return emitVectorDistributeConstraintsForOp(tunableOp, opAttr);
}

} // namespace mlir::iree_compiler
