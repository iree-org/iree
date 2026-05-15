// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

using AssertOp = IREE::Codegen::AssertOp;
using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;
using RootOpAttr = IREE::Codegen::RootOpAttr;

namespace {

/// Contraction-like dimension classification used by both matmul and conv.
struct ContractionLikeDims {
  SmallVector<unsigned> b;
  SmallVector<unsigned> m;
  SmallVector<unsigned> n;
  SmallVector<unsigned> k;
};

/// Problem size, loop count, and indexing maps for a root op.
struct RootOpLoopInfo {
  SmallVector<int64_t> staticLoopRanges;
  unsigned numLoops;
  SmallVector<AffineMap> indexingMaps;
};

// Keys for entries in the SMT constraints `knobs` dictionary.
// These names are aligned with lowering config and translation info fields.
constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobWorkgroupSizeKey = "workgroup_size";
constexpr StringLiteral kKnobSubgroupSizeKey = "subgroup_size";

// SMT variable names for knob values used in constraints.
constexpr StringLiteral kKnobMmaIdxName = "mma_idx";
constexpr StringLiteral kKnobSgMCntName = "sg_m_cnt";
constexpr StringLiteral kKnobSgNCntName = "sg_n_cnt";
constexpr StringLiteral kKnobSgSizeName = "sg_size";
constexpr StringLiteral kKnobWgSizeXName = "wg_size_x";
constexpr StringLiteral kKnobWgSizeYName = "wg_size_y";
constexpr StringLiteral kKnobWgSizeZName = "wg_size_z";

// SMT variable name prefixes. The loop dim count varies
// per problem, so names are built at runtime as prefix + dim idx.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";

// Default value for dims that are not knobbed.
constexpr int64_t kNoTileDimVal = 0;
constexpr int64_t kUnitTileDimVal = 1;

} // namespace

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

/// Get unique compatible MMA attrs for matmul and conv ops.
static SmallVector<Attribute>
getCompatibleMMAAttrs(linalg::LinalgOp op, IREE::GPU::TargetAttr gpuTarget,
                      const RootOpLoopInfo &loopInfo,
                      const ContractionLikeDims &dims) {
  if (gpuTarget.getWgp().getMma().empty()) {
    return {};
  }

  SmallVector<Attribute> mmaAttrs;
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();
  Type lhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  Type rhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(1)->get());
  Type initElemType = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  int64_t mSize = loopInfo.staticLoopRanges[dims.m.back()];
  int64_t nSize = loopInfo.staticLoopRanges[dims.n.back()];
  int64_t kSize = loopInfo.staticLoopRanges[dims.k.back()];

  // Dynamic shapes are not supported by tuner yet.
  if (ShapedType::isDynamic(mSize) || ShapedType::isDynamic(nSize) ||
      ShapedType::isDynamic(kSize)) {
    return {};
  }

  GPUMatmulShapeType problem{mSize,       nSize,       kSize,
                             lhsElemType, rhsElemType, initElemType};

  auto getIntrinsic = [](IREE::GPU::MMAAttr mma) -> GPUIntrinsicType {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  };

  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // VectorDistribute matmul/conv skip block intrinsics.
    if (mma.isBlockIntrinsic()) {
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
static FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims)) {
      return failure();
    }
    return ContractionLikeDims{llvm::to_vector(contractionDims->batch),
                               llvm::to_vector(contractionDims->m),
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
    return ContractionLikeDims{llvm::to_vector(convolutionDims->batch),
                               llvm::to_vector(convolutionDims->outputImage),
                               llvm::to_vector(convolutionDims->outputChannel),
                               llvm::to_vector(convolutionDims->inputChannel)};
  }
  return failure();
}

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

  // Build workgroup entries from lowering config semantics: untiled dims get 0,
  // batch and outer M/N dims get unit tile 1, and inner M/N dims are knobbed.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> unitWorkgroupDims;
  llvm::append_range(unitWorkgroupDims, dims.b);
  llvm::append_range(unitWorkgroupDims, dims.m);
  llvm::append_range(unitWorkgroupDims, dims.n);
  for (unsigned i : unitWorkgroupDims) {
    workgroupEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  // inferContractionLikeDims guarantees dims.m/n/k are non-empty for both
  // branches (asserted or early-returned for conv).
  workgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.m.back()));
  workgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));
  // Build reduction entries from the complement of unit workgroup dims.
  // Innermost K dim gets IntKnobAttr, outer K and filter dims get 1.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    if (llvm::is_contained(unitWorkgroupDims, i)) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

  // Build subgroup_basis as [[counts], [mapping]]. Mapping is an identity map
  // of const int for VectorDistribute matmul and conv. It is kept as a
  // placeholder so downstream constraint verification can match the
  // subgroup_basis knob template in the lowering config.
  // Only innermost M and N dims get subgroup tiling, others stay 1.
  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  subgroupCounts[dims.m.back()] = IntKnobAttr::get(ctx, kKnobSgMCntName);
  subgroupCounts[dims.n.back()] = IntKnobAttr::get(ctx, kKnobSgNCntName);
  SmallVector<Attribute> subgroupMapping;
  subgroupMapping.reserve(loopInfo.numLoops);
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  ArrayAttr subgroupBasis =
      ArrayAttr::get(ctx, {ArrayAttr::get(ctx, subgroupCounts),
                           ArrayAttr::get(ctx, subgroupMapping)});
  knobsEntries.emplace_back(kKnobSubgroupBasisKey, subgroupBasis);

  // Add workgroup size and subgroup size at the top level.
  SmallVector<Attribute> wgSizeKnobs = {
      IntKnobAttr::get(ctx, kKnobWgSizeXName),
      IntKnobAttr::get(ctx, kKnobWgSizeYName),
      IntKnobAttr::get(ctx, kKnobWgSizeZName)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

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
  unsigned mDim = dims.m.back();
  std::string mName = makeVarName(kKnobWgPrefix, mDim);
  Value wgTileMKnob = mkKnob(builder, loc, mName);
  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgTileMKnob,
      (kLoopRangePrefix + Twine(mDim) + " must be divisible by " + mName)
          .str());

  unsigned nDim = dims.n.back();
  std::string nName = makeVarName(kKnobWgPrefix, nDim);
  Value wgTileNKnob = mkKnob(builder, loc, nName);
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgTileNKnob,
      (kLoopRangePrefix + Twine(nDim) + " must be divisible by " + nName)
          .str());

  unsigned kDim = dims.k.back();
  std::string kName = makeVarName(kKnobRedPrefix, kDim);
  Value wgTileKKnob = mkKnob(builder, loc, kName);
  assertDivisible(
      builder, loc, smtDimArgs[kDim], wgTileKKnob,
      (kLoopRangePrefix + Twine(kDim) + " must be divisible by " + kName)
          .str());

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

  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  if (compatibleMMAs.empty()) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs =
      buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                         shell.smtDimArgs);
}

/// Multiple root ops may be present in a set, e.g. <set = 0>:
/// [linalg.fill, linalg.matmul]. This function will choose the matmul op
/// over the fill op.
static Operation *getTunableOp(ArrayRef<Operation *> rootOps) {
  if (rootOps.empty()) {
    return nullptr;
  }
  for (Operation *rootOp : rootOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
    if (linalgOp && (linalg::isaContractionOpInterface(linalgOp) ||
                     linalg::isaConvolutionOpInterface(linalgOp))) {
      return rootOp;
    }
    if (auto attnOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(rootOp)) {
      return rootOp;
    }
  }
  return nullptr;
}

LogicalResult emitLLVMGPUConstraints(Attribute attr,
                                     ArrayRef<Operation *> rootOps) {
  Operation *tunableOp = getTunableOp(rootOps);
  if (!tunableOp) {
    return success();
  }
  RootOpAttr opAttr = getRootOpInfo(tunableOp);
  if (!opAttr) {
    return success();
  }

  auto gpuPipelineAttr = cast<IREE::GPU::PipelineAttr>(attr);

  // Only VectorDistribute has constraint generation today.
  if (gpuPipelineAttr.getValue() !=
      IREE::GPU::LoweringPipeline::VectorDistribute) {
    return success();
  }

  return emitVectorDistributeConstraintsForOp(tunableOp, opAttr);
}

} // namespace mlir::iree_compiler
