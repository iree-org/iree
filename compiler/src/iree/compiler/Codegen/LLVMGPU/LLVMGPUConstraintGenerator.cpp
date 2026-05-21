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
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler {

using AssertOp = IREE::Codegen::AssertOp;
using LookupOp = IREE::Codegen::LookupOp;
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

// Lowering config knob list for VectorDistribute (VD):
//   workgroup = [wg_#, wg_#, ...], only innermost M and N dims are knobbed.
//   reduction = [..., ..., red_#], only innermost K dim is knobbed.
//   mma_kind = mma_idx
//   subgroup_basis = [[..., sg_m_cnt, sg_n_cnt, ...], [0, 1, 2, ...]]

// Lowering config knob list for TileAndFuse (TF):
//   workgroup = [wg_#, wg_#, ...], B, M, N dims are knobbed.
//   reduction = [..., ..., red_#], only innermost K dim is knobbed.
//   mma_kind = mma_idx
//   subgroup = [sg_#, sg_#, ...], M and N dims are knobbed.

// Translation info knob list (shared by both pipelines):
//   workgroup_size = [wg_size_x, wg_size_y, wg_size_z]
//   subgroup_size = sg_size

constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobSubgroupKey = "subgroup";
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

// SMT variable name prefixes.
// The loop dim count varies per problem, so names are built at runtime
// as prefix + dim idx, for example: wg_0, red_1, sg_0, etc.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";
constexpr StringLiteral kKnobSgPrefix = "sg_";

// Default value for dims that are not knobbed.
constexpr int64_t kNoTileDimVal = 0;
constexpr int64_t kUnitTileDimVal = 1;

} // namespace

// TODO(#23535): These constraints are derived from existing tuner constraints.
// Need to verify the validity of the additional heuristic constraints,
// such as `red_k % mma_m == 0`.

/// Assert: lhs % rhs == 0, with format args for diagnostics.
static void assertDivisible(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs, StringRef msg) {
  Value zero = mkIntConst(builder, loc, 0);
  Value rem = smt::IntModOp::create(builder, loc, lhs, rhs);
  Value eq = smt::EqOp::create(builder, loc, rem, zero);
  std::string fmtMsg = (msg + " ({} % {} == 0)").str();
  AssertOp::create(builder, loc, eq, fmtMsg, ValueRange{lhs, rhs});
}

/// Assert: lhs <pred> rhs
static void assertCmp(OpBuilder &builder, Location loc, smt::IntPredicate pred,
                      Value lhs, Value rhs, StringRef msg) {
  Value cmp = smt::IntCmpOp::create(builder, loc, pred, lhs, rhs);
  AssertOp::create(builder, loc, cmp, msg);
}

/// Assert: val >= lo && val <= hi.
static void assertBounds(OpBuilder &builder, Location loc, Value val,
                         StringRef name, Value lo, StringRef loName, Value hi,
                         StringRef hiName) {
  assertCmp(builder, loc, smt::IntPredicate::ge, val, lo,
            llvm::join_items("", name, " >= ", loName));
  assertCmp(builder, loc, smt::IntPredicate::le, val, hi,
            llvm::join_items("", name, " <= ", hiName));
}

/// Emit ceil(lhs / rhs) * rhs.
static Value emitAlignUpToMultiple(OpBuilder &builder, Location loc, Value lhs,
                                   Value rhs) {
  Value one = mkIntConst(builder, loc, 1);
  Value rhsMinusOne = smt::IntSubOp::create(builder, loc, rhs, one);
  Value lhsPlus =
      smt::IntAddOp::create(builder, loc, ValueRange{lhs, rhsMinusOne});
  Value div = smt::IntDivOp::create(builder, loc, lhsPlus, rhs);
  return smt::IntMulOp::create(builder, loc, ValueRange{div, rhs});
}

/// Emit product(values[0], values[1], ..., values[n-1]).
static Value emitProduct(OpBuilder &builder, Location loc,
                         ArrayRef<Value> values) {
  Value prod = mkIntConst(builder, loc, 1);
  for (Value value : values) {
    prod = smt::IntMulOp::create(builder, loc, ValueRange{prod, value});
  }
  return prod;
}

/// Get the LHS and RHS operand element-type byte sizes of a linalg op as SMT
/// int constants.
static std::pair<Value, Value>
getLhsRhsOperandBytes(OpBuilder &builder, Location loc,
                      linalg::LinalgOp linalgOp) {
  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  Value lhsBytes =
      mkIntConst(builder, loc, lhsType.getElementTypeBitWidth() / 8);
  Value rhsBytes =
      mkIntConst(builder, loc, rhsType.getElementTypeBitWidth() / 8);
  return {lhsBytes, rhsBytes};
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
inferContractionLikeDims(linalg::LinalgOp linalgOp,
                         IREE::GPU::LoweringPipeline pipeline) {
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
    if (failed(convolutionDims)) {
      return failure();
    }
    if (pipeline == IREE::GPU::LoweringPipeline::VectorDistribute) {
      if (convolutionDims->outputImage.empty() ||
          convolutionDims->outputChannel.empty() ||
          convolutionDims->inputChannel.empty()) {
        return failure();
      }
      // TODO(Amily): This mapping aligns with how VectorDistribute sets conv
      // dims. It may be too coarse for conv semantics and should be revisited
      // when conv constraint generation is fully plumbed.
      return ContractionLikeDims{
          llvm::to_vector(convolutionDims->batch),
          llvm::to_vector(convolutionDims->outputImage),
          llvm::to_vector(convolutionDims->outputChannel),
          llvm::to_vector(convolutionDims->inputChannel)};
    }
    if (pipeline == IREE::GPU::LoweringPipeline::TileAndFuse) {
      if (convolutionDims->outputChannel.empty() ||
          convolutionDims->inputChannel.empty() ||
          convolutionDims->filterLoop.empty() ||
          convolutionDims->outputImage.empty()) {
        return failure();
      }
      // Match TileAndFuse conv mapping.
      SmallVector<unsigned> mDims;
      llvm::append_range(mDims, convolutionDims->batch);
      llvm::append_range(mDims, convolutionDims->outputImage);
      llvm::sort(mDims);
      SmallVector<unsigned> kDims;
      llvm::append_range(kDims, convolutionDims->filterLoop);
      llvm::append_range(kDims, convolutionDims->inputChannel);
      return ContractionLikeDims{
          llvm::to_vector(convolutionDims->depth), mDims,
          llvm::to_vector(convolutionDims->outputChannel), kDims};
    }
    return failure();
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

/// Build the TileAndFuse knobs dict for contraction-like dims.
static DictionaryAttr
buildTileAndFuseKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                          const ContractionLikeDims &dims,
                          ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Build workgroup entries.
  // Batch, M, N dims are knobbed, K dims are not tiled and get 0.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> knobbedWorkgroupDims;
  llvm::append_range(knobbedWorkgroupDims, dims.b);
  llvm::append_range(knobbedWorkgroupDims, dims.m);
  llvm::append_range(knobbedWorkgroupDims, dims.n);
  for (unsigned i : knobbedWorkgroupDims) {
    workgroupEntries[i] = IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, i));
  }
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));

  // Build reduction entries.
  // Only innermost K dim is knobbed, other K dims are unit tiled as 1.
  // Other untiled dims get 0.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    reductionEntries[i] = makeIntAttr(ctx, kNoTileDimVal);
  }
  for (unsigned i : dims.k) {
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Build subgroup entries.
  // M and N dims are knobbed, other untiled dims get 0.
  SmallVector<Attribute> subgroupEntries(loopInfo.numLoops,
                                         makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> knobbedSubgroupDims;
  llvm::append_range(knobbedSubgroupDims, dims.m);
  llvm::append_range(knobbedSubgroupDims, dims.n);
  for (unsigned i : knobbedSubgroupDims) {
    subgroupEntries[i] = IntKnobAttr::get(ctx, makeVarName(kKnobSgPrefix, i));
  }
  knobsEntries.emplace_back(kKnobSubgroupKey,
                            ArrayAttr::get(ctx, subgroupEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

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

/// Emit smt.lookup ops to derive MMA m/n/k shape values from the mma_idx knob.
/// Returns {mmaMLookup, mmaNLookup, mmaKLookup} SSA values.
/// Also asserts bounds on mma_idx: 0 <= mma_idx <= N-1.
static std::tuple<Value, Value, Value>
emitMMALookup(OpBuilder &builder, Location loc, ArrayRef<Attribute> mmaAttrs) {

  Value mmaIdx = mkKnob(builder, loc, kKnobMmaIdxName);
  Value minMmaIdxVal = mkIntConst(builder, loc, 0);
  int64_t maxMmaIdx = static_cast<int64_t>(mmaAttrs.size()) - 1;
  Value maxMmaIdxVal = mkIntConst(builder, loc, maxMmaIdx);
  std::string maxMmaIdxName = Twine(maxMmaIdx).str();
  // Assert bounds: 0 <= mma_idx <= N-1.
  assertBounds(builder, loc, mmaIdx, kKnobMmaIdxName, minMmaIdxVal, "0",
               maxMmaIdxVal, maxMmaIdxName);

  // Build lookup tables for m, n, k.
  SmallVector<int64_t> keys, mVals, nVals, kVals;
  for (auto [i, attr] : llvm::enumerate(mmaAttrs)) {
    auto [m, n, k] = cast<IREE::GPU::MmaInterfaceAttr>(attr).getMNKShape();
    keys.push_back(static_cast<int64_t>(i));
    mVals.push_back(m);
    nVals.push_back(n);
    kVals.push_back(k);
  }
  auto intTy = smt::IntType::get(builder.getContext());
  return {LookupOp::create(builder, loc, intTy, mmaIdx, keys, mVals),
          LookupOp::create(builder, loc, intTy, mmaIdx, keys, nVals),
          LookupOp::create(builder, loc, intTy, mmaIdx, keys, kVals)};
}

/// Emit VectorDistribute constraints for contraction-like dims (matmul/conv).
static LogicalResult emitVectorDistributeConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  // Hardware constants.
  Value subgroupSizeVal =
      mkIntConst(builder, loc, gpuTarget.getPreferredSubgroupSize());
  Value maxThreadsVal = mkIntConst(
      builder, loc, gpuTarget.getWgp().getMaxThreadCountPerWorkgroup());
  Value maxSharedMemVal =
      mkIntConst(builder, loc, gpuTarget.getWgp().getMaxWorkgroupMemoryBytes());
  Value maxVGPRsVal = mkIntConst(builder, loc, 512);

  // Only innermost M, N, and K dims get constrained.
  unsigned mDim = dims.m.back();
  unsigned nDim = dims.n.back();
  unsigned kDim = dims.k.back();
  std::string wgMName = makeVarName(kKnobWgPrefix, mDim);
  std::string wgNName = makeVarName(kKnobWgPrefix, nDim);
  std::string redKName = makeVarName(kKnobRedPrefix, kDim);
  std::string mDimName = makeVarName(kLoopRangePrefix, mDim);
  std::string nDimName = makeVarName(kLoopRangePrefix, nDim);
  std::string kDimName = makeVarName(kLoopRangePrefix, kDim);

  // Create top-level knobs.
  Value wgM = mkKnob(builder, loc, wgMName);
  Value wgN = mkKnob(builder, loc, wgNName);
  Value redK = mkKnob(builder, loc, redKName);
  Value sgMCnt = mkKnob(builder, loc, kKnobSgMCntName);
  Value sgNCnt = mkKnob(builder, loc, kKnobSgNCntName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);
  Value wgSizeY = mkKnob(builder, loc, kKnobWgSizeYName);
  Value wgSizeZ = mkKnob(builder, loc, kKnobWgSizeZName);

  // Create intermediate variables.
  // Do not create these as knobs because they are not in the knob dict
  // and would fail constraint verification.
  // mma_m, mma_n, mma_k
  auto [mmaMLookup, mmaNLookup, mmaKLookup] =
      emitMMALookup(builder, loc, compatibleMMAs);
  // sg_num, number of subgroups per workgroup
  Value sgNum = smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  Value sgMDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, mmaMLookup});
  // sg_m, number of mma_m along M per subgroup
  Value sgM = smt::IntDivOp::create(builder, loc, wgM, sgMDenom);
  Value sgNDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, mmaNLookup});
  // sg_n, number of mma_n along N per subgroup
  Value sgN = smt::IntDivOp::create(builder, loc, wgN, sgNDenom);
  // sg_k, number of mma_k along K per subgroup
  Value sgK = smt::IntDivOp::create(builder, loc, redK, mmaKLookup);

  // Constraint 0: Set concrete values for knobs.
  // sg_size == preferred_subgroup_size
  Value sgSizeEq = smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal);
  AssertOp::create(builder, loc, sgSizeEq,
                   "sg_size == preferred_subgroup_size");

  // Constraint 1: Tile size must divide problem size.
  // dim % wg_tile == 0
  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgM,
      llvm::join_items("", mDimName, " must be divisible by ", wgMName));
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgN,
      llvm::join_items("", nDimName, " must be divisible by ", wgNName));
  assertDivisible(
      builder, loc, smtDimArgs[kDim], redK,
      llvm::join_items("", kDimName, " must be divisible by ", redKName));

  // Constraint 2: Tile size bounds.
  // mma <= wg_tile <= dim
  assertBounds(builder, loc, wgM, wgMName, mmaMLookup, "mma_m",
               smtDimArgs[mDim], mDimName);
  assertBounds(builder, loc, wgN, wgNName, mmaNLookup, "mma_n",
               smtDimArgs[nDim], nDimName);
  assertBounds(builder, loc, redK, redKName, mmaKLookup, "mma_k",
               smtDimArgs[kDim], kDimName);
  // wg_tile <= 512 (max VGPRs)
  assertCmp(builder, loc, smt::IntPredicate::le, wgM, maxVGPRsVal,
            llvm::join_items("", wgMName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, wgN, maxVGPRsVal,
            llvm::join_items("", wgNName, " <= 512 (max VGPRs)"));
  assertCmp(builder, loc, smt::IntPredicate::le, redK, maxVGPRsVal,
            llvm::join_items("", redKName, " <= 512 (max VGPRs)"));

  // Constraint 3: Tile size decomposition.
  // wg_tile % mma == 0
  assertDivisible(builder, loc, wgM, mmaMLookup,
                  llvm::join_items("", wgMName, " must be divisible by mma_m"));
  assertDivisible(builder, loc, wgN, mmaNLookup,
                  llvm::join_items("", wgNName, " must be divisible by mma_n"));
  assertDivisible(
      builder, loc, redK, mmaKLookup,
      llvm::join_items("", redKName, " must be divisible by mma_k"));
  // wg_m % (sg_m_cnt * mma_m * sg_m) == 0
  // wg_n % (sg_n_cnt * mma_n * sg_n) == 0
  // red_k % (1 * mma_k * sg_k) == 0
  Value mulResultM =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, mmaMLookup, sgM});
  Value mulResultN =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, mmaNLookup, sgN});
  Value mulResultK =
      smt::IntMulOp::create(builder, loc, ValueRange{mmaKLookup, sgK});
  assertDivisible(builder, loc, wgM, mulResultM,
                  llvm::join_items("", wgMName,
                                   " must be divisible by "
                                   "(sg_m_cnt * mma_m * sg_m)"));
  assertDivisible(builder, loc, wgN, mulResultN,
                  llvm::join_items("", wgNName,
                                   " must be divisible by "
                                   "(sg_n_cnt * mma_n * sg_n)"));

  assertDivisible(
      builder, loc, redK, mulResultK,
      llvm::join_items("", redKName, " must be divisible by (mma_k * sg_k)"));

  // Constraint 4: Subgroup count bounds.
  // wg_tile = mma * sg_tile * sg_cnt
  // Upper bound: max(wg_tile) / min(mma * 1) = 512 / 16 = 32.
  // 1 <= sg_m_cnt <= 32.
  // 1 <= sg_n_cnt <= 32.
  Value minSubgroupCntVal = mkIntConst(builder, loc, 1);
  Value maxSubgroupCntVal = mkIntConst(builder, loc, 32);
  assertBounds(builder, loc, sgMCnt, kKnobSgMCntName, minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgNCnt, kKnobSgNCntName, minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  // 1 <= sg_m <= 32.
  // 1 <= sg_n <= 32.
  // 1 <= sg_k <= 32.
  assertBounds(builder, loc, sgM, "sg_m", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgN, "sg_n", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  assertBounds(builder, loc, sgK, "sg_k", minSubgroupCntVal, "1",
               maxSubgroupCntVal, "32");
  // sg_m_cnt * sg_n_cnt == sg_num
  Value sgCntMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  Value sgCntEq = smt::EqOp::create(builder, loc, sgCntMulResult, sgNum);
  AssertOp::create(builder, loc, sgCntEq, "sg_m_cnt * sg_n_cnt == sg_num");
  // TODO(#23535): bounds range copied from current Tuner. Need to revisit.
  // 1 <= sg_num <= 10.
  assertBounds(builder, loc, sgNum, "sg_num", mkIntConst(builder, loc, 1), "1",
               mkIntConst(builder, loc, 10), "10");

  // Constraint 5: Thread count limit.
  // sg_m_cnt * sg_n_cnt * sg_size <= max_threads
  Value totalThreadsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreadsMulResult,
            maxThreadsVal, "total_threads <= max_threads");

  // Constraint 6: Load distribution.
  // (red_k * wg_m) % (sg_m_cnt * sg_n_cnt * sg_size) == 0
  // (red_k * wg_n) % (sg_m_cnt * sg_n_cnt * sg_size) == 0
  auto emitLoadDistForWg = [&](Value wg, StringRef name) -> void {
    Value lhsMulResult =
        smt::IntMulOp::create(builder, loc, ValueRange{redK, wg});
    Value rhsMulResult =
        smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
    assertDivisible(
        builder, loc, lhsMulResult, rhsMulResult,
        llvm::join_items("", name, " must be divisible by thread_count"));
  };
  emitLoadDistForWg(wgM, "lhs_tile_elements");
  emitLoadDistForWg(wgN, "rhs_tile_elements");

  // Constraint 7: Shared memory limit.
  // Approximate formula:
  // (lhs_bytes * wg_m * red_k) + (rhs_bytes * wg_n * red_k)
  // Get element type bitwidths from operands.
  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  int64_t lhsBytes = lhsType.getElementTypeBitWidth() / 8;
  int64_t rhsBytes = rhsType.getElementTypeBitWidth() / 8;
  Value lhsBytesVal = mkIntConst(builder, loc, lhsBytes);
  Value rhsBytesVal = mkIntConst(builder, loc, rhsBytes);

  Value lhsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{lhsBytesVal, wgM, redK});
  Value rhsMulResult =
      smt::IntMulOp::create(builder, loc, ValueRange{rhsBytesVal, wgN, redK});
  Value totalSharedMem = smt::IntAddOp::create(
      builder, loc, ValueRange{lhsMulResult, rhsMulResult});
  assertCmp(builder, loc, smt::IntPredicate::le, totalSharedMem,
            maxSharedMemVal, "shared memory must fit in workgroup memory");

  // Constraint 8: TranslationInfo workgroup structure.
  // For VectorDistribute, workgroup_size =
  // [wg_x, wg_y, wg_z] = [total_threads, 1, 1]
  // wg_x == total_threads
  Value wgXEq = smt::EqOp::create(builder, loc, wgSizeX, totalThreadsMulResult);
  AssertOp::create(builder, loc, wgXEq, "wg_size_x == total_threads");
  // wg_y == 1
  Value wgYEq =
      smt::EqOp::create(builder, loc, wgSizeY, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgYEq, "wg_size_y == 1");
  // wg_z == 1
  Value wgZEq =
      smt::EqOp::create(builder, loc, wgSizeZ, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgZEq, "wg_size_z == 1");

  // Constraint 9: Additional heuristic filters to narrow the search space.
  // wg_x <= wg_m OR wg_x <= wg_n
  Value wgXLeM =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, wgSizeX, wgM);
  Value wgXLeN =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, wgSizeX, wgN);
  Value orCond = smt::OrOp::create(builder, loc, ValueRange{wgXLeM, wgXLeN});
  AssertOp::create(builder, loc, orCond,
                   "wg_size_x <= wg_m OR wg_size_x <= wg_n");
  // red_k % mma_m == 0
  assertDivisible(
      builder, loc, redK, mmaMLookup,
      llvm::join_items("", redKName, " must be divisible by mma_m"));

  return success();
}

/// Emit TileAndFuse constraints for contraction-like dims (matmul/conv).
static LogicalResult emitTileAndFuseConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  // Hardware constants.
  Value subgroupSizeVal =
      mkIntConst(builder, loc, gpuTarget.getPreferredSubgroupSize());
  Value maxThreadsVal = mkIntConst(
      builder, loc, gpuTarget.getWgp().getMaxThreadCountPerWorkgroup());
  Value maxSharedMemVal =
      mkIntConst(builder, loc, gpuTarget.getWgp().getMaxWorkgroupMemoryBytes());
  Value maxVGPRsVal = mkIntConst(builder, loc, 512);

  // Innermost M, N, K dims.
  unsigned mInnerDim = dims.m.back();
  unsigned nInnerDim = dims.n.back();
  unsigned kInnerDim = dims.k.back();
  std::string wgMInnerName = makeVarName(kKnobWgPrefix, mInnerDim);
  std::string wgNInnerName = makeVarName(kKnobWgPrefix, nInnerDim);
  std::string redKInnerName = makeVarName(kKnobRedPrefix, kInnerDim);
  std::string kInnerDimName = makeVarName(kLoopRangePrefix, kInnerDim);
  std::string sgMInnerName = makeVarName(kKnobSgPrefix, mInnerDim);
  std::string sgNInnerName = makeVarName(kKnobSgPrefix, nInnerDim);

  // Create top-level knobs from lowering and translation configs.
  SmallVector<unsigned> wgMNBDims;
  llvm::append_range(wgMNBDims, dims.m);
  llvm::append_range(wgMNBDims, dims.n);
  llvm::append_range(wgMNBDims, dims.b);
  SmallVector<unsigned> sgMNDims;
  llvm::append_range(sgMNDims, dims.m);
  llvm::append_range(sgMNDims, dims.n);
  auto createKnobsByDim = [&](ArrayRef<unsigned> dimList,
                              StringRef knobPrefix) -> SmallVector<Value> {
    SmallVector<Value> knobsByDim(smtDimArgs.size());
    for (unsigned dim : dimList) {
      knobsByDim[dim] = mkKnob(builder, loc, makeVarName(knobPrefix, dim));
    }
    return knobsByDim;
  };
  SmallVector<Value> wgKnobsByDim = createKnobsByDim(wgMNBDims, kKnobWgPrefix);
  SmallVector<Value> sgKnobsByDim = createKnobsByDim(sgMNDims, kKnobSgPrefix);
  Value wgMInner = wgKnobsByDim[mInnerDim];
  Value wgNInner = wgKnobsByDim[nInnerDim];
  Value sgM = sgKnobsByDim[mInnerDim];
  Value sgN = sgKnobsByDim[nInnerDim];
  Value redK = mkKnob(builder, loc, redKInnerName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);
  Value wgSizeY = mkKnob(builder, loc, kKnobWgSizeYName);
  Value wgSizeZ = mkKnob(builder, loc, kKnobWgSizeZName);

  // Create intermediate variables.
  // Do not create these as knobs because they are not in the knob dict
  // and would fail constraint verification.
  // mma_m, mma_n, mma_k
  auto [mmaMLookup, mmaNLookup, mmaKLookup] =
      emitMMALookup(builder, loc, compatibleMMAs);
  Value sgMDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgM, mmaMLookup});
  Value sgNDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgN, mmaNLookup});
  // sg_m_cnt = wg_m / (sg_m * mma_m)
  Value sgMCnt = smt::IntDivOp::create(builder, loc, wgMInner, sgMDenom);
  // sg_n_cnt = wg_n / (sg_n * mma_n)
  Value sgNCnt = smt::IntDivOp::create(builder, loc, wgNInner, sgNDenom);
  // sg_num = sg_m_cnt * sg_n_cnt
  Value subgroups =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  // total_threads = sg_m_cnt * sg_n_cnt * sg_size
  Value totalThreads =
      smt::IntMulOp::create(builder, loc, ValueRange{subgroups, sgSize});

  // Constraint 0: Set concrete values for knobs.
  // sg_size == preferred_subgroup_size.
  Value sgSizeEq = smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal);
  AssertOp::create(builder, loc, sgSizeEq,
                   "sg_size == preferred_subgroup_size");

  // Constraint 1: Workgroup Tiles.
  // Basic constraints for both outer and innermost M and N tiles.
  // TODO: Add overpadding to problem M and N dims to get better tile sizes.
  for (unsigned dim : wgMNBDims) {
    std::string wgName = makeVarName(kKnobWgPrefix, dim);
    std::string problemName = makeVarName(kLoopRangePrefix, dim);
    std::string sgName = makeVarName(kKnobSgPrefix, dim);
    Value wg = wgKnobsByDim[dim];
    Value sg = sgKnobsByDim[dim];
    // dim % wg == 0.
    assertDivisible(
        builder, loc, smtDimArgs[dim], wg,
        llvm::join_items("", problemName, " must be divisible by ", wgName));
    // 1 <= wg <= dim
    assertBounds(builder, loc, wg, wgName, mkIntConst(builder, loc, 1), "1",
                 smtDimArgs[dim], problemName);
    // wg % sg == 0.
    assertDivisible(
        builder, loc, wg, sg,
        llvm::join_items("", wgName, " must be divisible by ", sgName));
  }
  // Extra constraints for innermost M and N tiles.
  // wg >= mma
  assertCmp(builder, loc, smt::IntPredicate::ge, wgMInner, mmaMLookup,
            llvm::join_items("", wgMInnerName, " >= mma_m"));
  assertCmp(builder, loc, smt::IntPredicate::ge, wgNInner, mmaNLookup,
            llvm::join_items("", wgNInnerName, " >= mma_n"));
  // wg % mma == 0.
  assertDivisible(
      builder, loc, wgMInner, mmaMLookup,
      llvm::join_items("", wgMInnerName, " must be divisible by mma_m"));
  assertDivisible(
      builder, loc, wgNInner, mmaNLookup,
      llvm::join_items("", wgNInnerName, " must be divisible by mma_n"));
  // wg % (sg * mma) == 0
  Value mInnerDiv =
      smt::IntMulOp::create(builder, loc, ValueRange{sgM, mmaMLookup});
  Value nInnerDiv =
      smt::IntMulOp::create(builder, loc, ValueRange{sgN, mmaNLookup});
  assertDivisible(builder, loc, wgMInner, mInnerDiv,
                  llvm::join_items("", wgMInnerName, " must be divisible by ",
                                   sgMInnerName, " * mma_m"));
  assertDivisible(builder, loc, wgNInner, nInnerDiv,
                  llvm::join_items("", wgNInnerName, " must be divisible by ",
                                   sgNInnerName, " * mma_n"));
  // Product constraints for all M and N tiles.
  // prod(wg) <= max_vgprs
  auto buildWgProd = [&](ArrayRef<unsigned> axisDims) -> Value {
    SmallVector<Value> values;
    for (unsigned dim : axisDims) {
      values.push_back(wgKnobsByDim[dim]);
    }
    return emitProduct(builder, loc, values);
  };
  Value wgMProd = buildWgProd(dims.m);
  Value wgNProd = buildWgProd(dims.n);
  assertCmp(builder, loc, smt::IntPredicate::le, wgMProd, maxVGPRsVal,
            "prod(wg_m) <= 512 (max_vgprs)");
  assertCmp(builder, loc, smt::IntPredicate::le, wgNProd, maxVGPRsVal,
            "prod(wg_n) <= 512 (max_vgprs)");

  // prod(wg) == prod(sg * sg_cnt * mma)
  auto buildSgProd = [&](ArrayRef<unsigned> axisDims) -> Value {
    SmallVector<Value> values;
    for (unsigned dim : axisDims) {
      values.push_back(sgKnobsByDim[dim]);
    }
    return emitProduct(builder, loc, values);
  };
  Value allSgMProd = buildSgProd(dims.m);
  Value allSgNProd = buildSgProd(dims.n);
  Value mRhsProd = smt::IntMulOp::create(
      builder, loc, ValueRange{allSgMProd, sgMCnt, mmaMLookup});
  Value nRhsProd = smt::IntMulOp::create(
      builder, loc, ValueRange{allSgNProd, sgNCnt, mmaNLookup});
  Value mProdEq = smt::EqOp::create(builder, loc, wgMProd, mRhsProd);
  Value nProdEq = smt::EqOp::create(builder, loc, wgNProd, nRhsProd);
  AssertOp::create(builder, loc, mProdEq,
                   "prod(wg_m) == prod(sg_m) * sg_m_cnt * mma_m");
  AssertOp::create(builder, loc, nProdEq,
                   "prod(wg_n) == prod(sg_n) * sg_n_cnt * mma_n");
  // Batch dim tile constraints.
  // TODO: This constraint can be expanded to allow wg_b values other than 1.
  // wg_b == 1
  for (auto dim : dims.b) {
    std::string wgBatchName = makeVarName(kKnobWgPrefix, dim);
    Value wgBatch = wgKnobsByDim[dim];
    Value batchEqOne =
        smt::EqOp::create(builder, loc, wgBatch, mkIntConst(builder, loc, 1));
    AssertOp::create(builder, loc, batchEqOne,
                     llvm::join_items("", wgBatchName, " == 1"));
  }

  // Constraint 2: Reduction Tile.
  // aligned_dim_k % red_k == 0, aligned_dim_k = ceil(dim_k / mma_k) * mma_k
  // problemInnerKAligned = ceil(dim_k / mma_k) * mma_k.
  Value problemInnerKAligned =
      emitAlignUpToMultiple(builder, loc, smtDimArgs[kInnerDim], mmaKLookup);
  assertDivisible(builder, loc, problemInnerKAligned, redK,
                  llvm::join_items("", "aligned inner k dim ", kInnerDimName,
                                   " must be divisible by ", redKInnerName));
  // red_k <= max_vgprs
  assertCmp(builder, loc, smt::IntPredicate::le, redK, maxVGPRsVal,
            llvm::join_items("", redKInnerName, " <= 512 (max_vgprs)"));
  // red_k <= aligned_dim_k
  assertCmp(builder, loc, smt::IntPredicate::le, redK, problemInnerKAligned,
            llvm::join_items("", redKInnerName, " <= aligned k dim"));
  // unpacked_k >= 1, unpacked_k = red_k / mma_k
  Value unpackedK = smt::IntDivOp::create(builder, loc, redK, mmaKLookup);
  assertCmp(builder, loc, smt::IntPredicate::ge, unpackedK,
            mkIntConst(builder, loc, 1),
            llvm::join_items("", "1 <= ", redKInnerName, " / mma_k"));

  // Constraint 3: Subgroup tile size.
  // sg >= 1 for M and N tiles.
  for (unsigned dim : sgMNDims) {
    std::string sgName = makeVarName(kKnobSgPrefix, dim);
    Value sg = sgKnobsByDim[dim];
    assertCmp(builder, loc, smt::IntPredicate::ge, sg,
              mkIntConst(builder, loc, 1),
              llvm::join_items("", sgName, " >= 1"));
  }
  // 1 <= sg_cnt <= 32
  assertBounds(builder, loc, sgMCnt, "sg_m_cnt", mkIntConst(builder, loc, 1),
               "1", mkIntConst(builder, loc, 32), "32");
  assertBounds(builder, loc, sgNCnt, "sg_n_cnt", mkIntConst(builder, loc, 1),
               "1", mkIntConst(builder, loc, 32), "32");
  // 1 <= sg_num <= 10
  assertBounds(builder, loc, subgroups, "sg_num", mkIntConst(builder, loc, 1),
               "1", mkIntConst(builder, loc, 10), "10");

  // Constraint 4: Thread count limit.
  // sg_m_cnt * sg_n_cnt * sg_size <= max_threads.
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreads, maxThreadsVal,
            "total_threads <= max_threads");

  // Constraint 5: Shared memory limit.
  // (lhs_bytes * wg_m * red_k) + (rhs_bytes * wg_n * red_k)
  auto [lhsBytesVal, rhsBytesVal] =
      getLhsRhsOperandBytes(builder, loc, linalgOp);
  Value lhsMem = smt::IntMulOp::create(builder, loc,
                                       ValueRange{lhsBytesVal, wgMInner, redK});
  Value rhsMem = smt::IntMulOp::create(builder, loc,
                                       ValueRange{rhsBytesVal, wgNInner, redK});
  Value totalSharedMem =
      smt::IntAddOp::create(builder, loc, ValueRange{lhsMem, rhsMem});
  assertCmp(builder, loc, smt::IntPredicate::le, totalSharedMem,
            maxSharedMemVal, "shared memory must fit in workgroup memory");

  // Constraint 6: TranslationInfo workgroup structure.
  // For TileAndFuse, workgroup_size = [total_threads, 1, 1].
  Value wgXEq = smt::EqOp::create(builder, loc, wgSizeX, totalThreads);
  AssertOp::create(builder, loc, wgXEq, "wg_size_x == total_threads");
  Value wgYEq =
      smt::EqOp::create(builder, loc, wgSizeY, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgYEq, "wg_size_y == 1");
  Value wgZEq =
      smt::EqOp::create(builder, loc, wgSizeZ, mkIntConst(builder, loc, 1));
  AssertOp::create(builder, loc, wgZEq, "wg_size_z == 1");

  return success();
}

/// Emit constraints for a single root op under a supported LLVMGPU pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult
emitConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr,
                     IREE::GPU::LoweringPipeline pipeline) {
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

  FailureOr<ContractionLikeDims> dims =
      inferContractionLikeDims(linalgOp, pipeline);
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
  DictionaryAttr knobs;
  if (pipeline == IREE::GPU::LoweringPipeline::VectorDistribute) {
    knobs =
        buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  } else if (pipeline == IREE::GPU::LoweringPipeline::TileAndFuse) {
    knobs = buildTileAndFuseKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  } else {
    return success();
  }
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(ctx, pipeline);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  if (pipeline == IREE::GPU::LoweringPipeline::VectorDistribute) {
    return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                           shell.smtDimArgs, compatibleMMAs);
  }
  if (pipeline == IREE::GPU::LoweringPipeline::TileAndFuse) {
    return emitTileAndFuseConstraints(builder, linalgOp, *dims, gpuTarget,
                                      shell.smtDimArgs, compatibleMMAs);
  }
  return success();
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

  IREE::GPU::LoweringPipeline pipeline = gpuPipelineAttr.getValue();
  if (pipeline != IREE::GPU::LoweringPipeline::VectorDistribute &&
      pipeline != IREE::GPU::LoweringPipeline::TileAndFuse) {
    return success();
  }
  return emitConstraintsForOp(tunableOp, opAttr, pipeline);
}

} // namespace mlir::iree_compiler
