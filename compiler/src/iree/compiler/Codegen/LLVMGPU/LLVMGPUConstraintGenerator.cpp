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
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-constraint-generator"
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

/// Optional TileAndFuse constraint inputs for IGEMM convolution.
struct TileAndFuseConstraintOptions {
  bool useIgemm = false;
  ArrayRef<unsigned> fusedIgemmKDims = {};
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

// Translation info knob list (TileAndFuse only):
//   use_igemm_convolution = use_igemm_convolution ("true" or "false")

constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobSubgroupKey = "subgroup";
constexpr StringLiteral kKnobWorkgroupSizeKey = "workgroup_size";
constexpr StringLiteral kKnobSubgroupSizeKey = "subgroup_size";
constexpr StringLiteral kKnobUseIgemmConvolutionKey = "use_igemm_convolution";

// SMT variable names for knob values used in constraints.
constexpr StringLiteral kKnobMmaIdxName = "mma_idx";
constexpr StringLiteral kKnobSgMCntName = "sg_m_cnt";
constexpr StringLiteral kKnobSgNCntName = "sg_n_cnt";
constexpr StringLiteral kKnobSgSizeName = "sg_size";
constexpr StringLiteral kKnobWgSizeXName = "wg_size_x";
constexpr StringLiteral kKnobWgSizeYName = "wg_size_y";
constexpr StringLiteral kKnobWgSizeZName = "wg_size_z";
constexpr StringLiteral kKnobUseIgemmConvolutionName = "use_igemm_convolution";

// SMT variable name prefixes.
// The loop dim count varies per problem, so names are built at runtime
// as prefix + dim idx, for example: wg_0, red_1, sg_0, etc.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";
constexpr StringLiteral kKnobSgPrefix = "sg_";

// Default value for dims that are not knobbed.
constexpr int64_t kNoTileDimVal = 0;
constexpr int64_t kUnitTileDimVal = 1;

//===-------------------------------------------------------------------===//
// Attention-specific keys + knob names. Attention has two matmuls (QK and PV),
// so MMA selection is tracked separately for each decomposition leg.
//===-------------------------------------------------------------------===//

// Top-level attention lowering_config keys (mirrors native attention
// codegen output from KernelConfig.cpp's setAttention...: `workgroup`,
// `reduction`, `promote_operands` + `promotion_types`,
// `decomposition_config`).
constexpr StringLiteral kKnobPromoteOperandsKey = "promote_operands";
// `kDecompositionConfigKey` lives in IREECodegenAttrs.h so the
// materializer reads back the same string this emitter writes.

// Decomposition-config nested keys (per OnlineAttentionOp's getQKAttrStr /
// getPVAttrStr -- verbatim "qk_attrs" / "pv_attrs").
constexpr StringLiteral kKnobQKAttrsKey = "qk_attrs";
constexpr StringLiteral kKnobPVAttrsKey = "pv_attrs";
constexpr StringLiteral kKnobLoweringConfigKey = "lowering_config";

// Per-matmul mma_idx knob names (one_of_knob over compatible MMAs).
constexpr StringLiteral kKnobQKMmaIdxName = "qk_mma_idx";
constexpr StringLiteral kKnobPVMmaIdxName = "pv_mma_idx";

// Top-level tile knob names (the M/N/K of the attention iteration domain;
// here K refers to K2, the key sequence length, per AttentionOpDetail).
constexpr StringLiteral kKnobAttnMTileName = "m_tile";
constexpr StringLiteral kKnobAttnNTileName = "n_tile";
constexpr StringLiteral kKnobAttnK2TileName = "red_k2";

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

/// Emit tuner-style overpadded bound for a dimension.
/// If dim > 128: align up to 128. Else if dim > 32: align up to 32.
/// Otherwise keep dim unchanged.
static Value emitOverpaddedBound(OpBuilder &builder, Location loc, Value dim) {
  Value c32 = mkIntConst(builder, loc, 32);
  Value c33 = mkIntConst(builder, loc, 33);
  Value c128 = mkIntConst(builder, loc, 128);
  Value c129 = mkIntConst(builder, loc, 129);
  Value align32 = emitAlignUpToMultiple(builder, loc, dim, c32);
  Value align128 = emitAlignUpToMultiple(builder, loc, dim, c128);
  Value gt32 =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, dim, c33);
  Value gt128 =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, dim, c129);
  Value padded32 = smt::IteOp::create(builder, loc, gt32, align32, dim);
  return smt::IteOp::create(builder, loc, gt128, align128, padded32);
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

/// Assert: lhs == rhs.
static void assertEq(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                     StringRef msg) {
  Value eq = smt::EqOp::create(builder, loc, lhs, rhs);
  AssertOp::create(builder, loc, eq, msg);
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

/// Returns original conv K indices that fuse in IGEMM, excluding innermost.
/// For conv K [3, 4, 5] mapping to [3, 3, 3], returns [3, 4]. For conv K
/// [1, 4, 5] mapping to [1, 4, 4], returns [4].
static SmallVector<unsigned>
getFusedIgemmKDimIndices(linalg::LinalgOp linalgOp,
                         ArrayRef<unsigned> convKDims) {
  if (convKDims.empty()) {
    return {};
  }

  FailureOr<IREE::LinalgExt::IGEMMGenericConvDetails> igemmDetails =
      IREE::LinalgExt::getIGEMMGenericConvDetails(linalgOp);
  if (failed(igemmDetails)) {
    return {};
  }

  auto mapConvDimToIgemm = [&](unsigned convDim) -> unsigned {
    return cast<AffineDimExpr>(igemmDetails->convToIgemmDimMap.lookup(convDim))
        .getPosition();
  };

  DenseMap<unsigned, unsigned> igemmDimToConvKCount;
  for (unsigned convKDim : convKDims) {
    ++igemmDimToConvKCount[mapConvDimToIgemm(convKDim)];
  }

  unsigned innermostConvK = convKDims.back();
  SmallVector<unsigned> fusedConvKDims;
  for (unsigned convKDim : convKDims) {
    if (convKDim == innermostConvK) {
      continue;
    }
    if (igemmDimToConvKCount[mapConvDimToIgemm(convKDim)] > 1) {
      fusedConvKDims.push_back(convKDim);
    }
  }
  return fusedConvKDims;
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
/// When `useIgemm` is true, also adds the `use_igemm_convolution` one-of
/// knob entry with string options "false"/"true".
static DictionaryAttr buildTileAndFuseKnobsDict(
    MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
    const ContractionLikeDims &dims, ArrayRef<Attribute> compatibleMMAs,
    bool isConv = false, ArrayRef<unsigned> fusedIgemmKDims = {}) {
  SmallVector<NamedAttribute> knobsEntries;

  unsigned layoutNumLoops = loopInfo.numLoops;
  if (!fusedIgemmKDims.empty()) {
    layoutNumLoops = loopInfo.numLoops - fusedIgemmKDims.size();
  }

  // Build workgroup entries.
  // Batch, M, N dims are knobbed, K dims are not tiled and get 0.
  SmallVector<Attribute> workgroupEntries(layoutNumLoops,
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
  SmallVector<Attribute> reductionEntries(layoutNumLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (auto i : knobbedWorkgroupDims) {
    reductionEntries[i] = makeIntAttr(ctx, kNoTileDimVal);
  }
  for (unsigned i : dims.k) {
    if (llvm::is_contained(fusedIgemmKDims, i) || i >= layoutNumLoops) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[layoutNumLoops - 1] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Build subgroup entries.
  // M and N dims are knobbed, other untiled dims get 0.
  SmallVector<Attribute> subgroupEntries(layoutNumLoops,
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

  // Conv-only: use_igemm_convolution one-of knob with options "false"/"true".
  if (isConv) {
    knobsEntries.emplace_back(
        kKnobUseIgemmConvolutionKey,
        OneOfKnobAttr::get(
            ctx, kKnobUseIgemmConvolutionName,
            {StringAttr::get(ctx, "false"), StringAttr::get(ctx, "true")}));
  }

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit smt.lookup ops to derive MMA m/n/k shape values from the mma_idx knob.
/// Returns {mmaIdxKnob, mmaMLookup, mmaNLookup, mmaKLookup} SSA values.
/// The first element is the MMA-index knob itself so attention's
/// downstream layout-binding can re-use it (the verifier rejects
/// duplicate KnobOps for the same knob name).
/// Also asserts bounds on the MMA index knob: 0 <= idx <= N-1.
/// `mmaIdxKnobName` parameterizes the knob name so attention (which has
/// two independent MMA selections -- QK and PV) can call this twice with
/// distinct names ("qk_mma_idx" / "pv_mma_idx") and not collide on
/// "mma_idx".
static std::tuple<Value, Value, Value, Value>
emitMMALookup(OpBuilder &builder, Location loc, ArrayRef<Attribute> mmaAttrs,
              StringRef mmaIdxKnobName = kKnobMmaIdxName) {
  Value mmaIdx = mkKnob(builder, loc, mmaIdxKnobName);
  Value minMmaIdxVal = mkIntConst(builder, loc, 0);
  int64_t maxMmaIdx = static_cast<int64_t>(mmaAttrs.size()) - 1;
  Value maxMmaIdxVal = mkIntConst(builder, loc, maxMmaIdx);
  std::string maxMmaIdxName = Twine(maxMmaIdx).str();
  // Assert bounds: 0 <= idx <= N-1.
  assertBounds(builder, loc, mmaIdx, mmaIdxKnobName, minMmaIdxVal, "0",
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
  return {mmaIdx, LookupOp::create(builder, loc, intTy, mmaIdx, keys, mVals),
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
  // mma_m, mma_n, mma_k. The first tuple element (the mma_idx knob
  // itself) is unused on the matmul path -- only attention needs to
  // re-use it for the layout-binding lookups.
  auto [unusedMmaIdx, mmaMLookup, mmaNLookup, mmaKLookup] =
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
/// When `options.useIgemm` is true, also emits the `use_igemm_convolution`
/// one-of knob and bounds constraints on its option index.
/// `options.fusedIgemmKDims` lists original conv K indices that fuse in
/// IGEMM, excluding innermost.
static LogicalResult emitTileAndFuseConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs,
    TileAndFuseConstraintOptions options = {}) {
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
  // mma_m, mma_n, mma_k. The first tuple element (the mma_idx knob
  // itself) is unused on the TileAndFuse matmul path -- only attention
  // needs to re-use it for the layout-binding lookups.
  auto [unusedMmaIdx, mmaMLookup, mmaNLookup, mmaKLookup] =
      emitMMALookup(builder, loc, compatibleMMAs);
  auto buildWgProd = [&](ArrayRef<unsigned> axisDims) -> Value {
    SmallVector<Value> values;
    for (unsigned dim : axisDims) {
      values.push_back(wgKnobsByDim[dim]);
    }
    return emitProduct(builder, loc, values);
  };
  auto buildSgProd = [&](ArrayRef<unsigned> axisDims) -> Value {
    SmallVector<Value> values;
    for (unsigned dim : axisDims) {
      values.push_back(sgKnobsByDim[dim]);
    }
    return emitProduct(builder, loc, values);
  };
  Value wgMProd = buildWgProd(dims.m);
  Value wgNProd = buildWgProd(dims.n);
  Value allSgMProd = buildSgProd(dims.m);
  Value allSgNProd = buildSgProd(dims.n);
  Value sgMDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{allSgMProd, mmaMLookup});
  Value sgNDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{allSgNProd, mmaNLookup});
  smt::IntType smtIntTy = smt::IntType::get(builder.getContext());
  auto sgMCntDecl = smt::DeclareFunOp::create(
      builder, loc, smtIntTy, builder.getStringAttr(kKnobSgMCntName));
  auto sgNCntDecl = smt::DeclareFunOp::create(
      builder, loc, smtIntTy, builder.getStringAttr(kKnobSgNCntName));
  Value sgMCnt = sgMCntDecl.getResult();
  Value sgNCnt = sgNCntDecl.getResult();
  // sg_num = sg_m_cnt * sg_n_cnt
  Value subgroups =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  // total_threads = sg_m_cnt * sg_n_cnt * sg_size
  Value totalThreads =
      smt::IntMulOp::create(builder, loc, ValueRange{subgroups, sgSize});
  // packed_k = red_k * mma_k
  Value packedK =
      smt::IntMulOp::create(builder, loc, ValueRange{redK, mmaKLookup});
  // problemInnerAligned = ceil(dim / mma) * mma.
  Value problemInnerMAligned =
      emitAlignUpToMultiple(builder, loc, smtDimArgs[mInnerDim], mmaMLookup);
  Value problemInnerNAligned =
      emitAlignUpToMultiple(builder, loc, smtDimArgs[nInnerDim], mmaNLookup);
  // problemInnerKAligned = ceil(dim_k / mma_k) * mma_k, or dim_k * dim_fused
  // with IGEMM.
  Value alignedInnerK =
      emitAlignUpToMultiple(builder, loc, smtDimArgs[kInnerDim], mmaKLookup);
  Value problemInnerKAligned = alignedInnerK;
  Value useIgemmTrue;
  if (options.useIgemm) {
    // use_igemm_convolution knob, 0 -> direct conv, 1 -> IGEMM conv
    Value useIgemmIdx = mkKnob(builder, loc, kKnobUseIgemmConvolutionName);
    assertBounds(builder, loc, useIgemmIdx, kKnobUseIgemmConvolutionName,
                 mkIntConst(builder, loc, 0), "0", mkIntConst(builder, loc, 1),
                 "1");

    SmallVector<Value> fusedDimValues;
    fusedDimValues.reserve(options.fusedIgemmKDims.size());
    for (unsigned dim : options.fusedIgemmKDims) {
      fusedDimValues.push_back(smtDimArgs[dim]);
    }
    Value dimFused = emitProduct(builder, loc, fusedDimValues);
    Value igemmInnerK = smt::IntMulOp::create(
        builder, loc, ValueRange{smtDimArgs[kInnerDim], dimFused});

    useIgemmTrue = smt::EqOp::create(builder, loc, useIgemmIdx,
                                     mkIntConst(builder, loc, 1));
    problemInnerKAligned = smt::IteOp::create(builder, loc, useIgemmTrue,
                                              igemmInnerK, alignedInnerK);
  }

  // Constraint 0: Set concrete values for knobs.
  // sg_size == preferred_subgroup_size.
  Value sgSizeEq = smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal);
  AssertOp::create(builder, loc, sgSizeEq,
                   "sg_size == preferred_subgroup_size");

  // Constraint 1: Workgroup Tiles.
  // Basic constraints for both outer and innermost M and N tiles.
  for (unsigned dim : wgMNBDims) {
    std::string wgName = makeVarName(kKnobWgPrefix, dim);
    std::string problemName = makeVarName(kLoopRangePrefix, dim);
    std::string sgName = makeVarName(kKnobSgPrefix, dim);
    Value wg = wgKnobsByDim[dim];
    Value sg = sgKnobsByDim[dim];
    Value problemDim = smtDimArgs[dim];
    if (dim == mInnerDim) {
      problemDim = problemInnerMAligned;
    } else if (dim == nInnerDim) {
      problemDim = problemInnerNAligned;
    } else if (options.useIgemm && (llvm::is_contained(dims.m, dim) ||
                                    llvm::is_contained(dims.n, dim))) {
      Value overpaddedDim = emitOverpaddedBound(builder, loc, problemDim);
      problemDim = smt::IteOp::create(builder, loc, useIgemmTrue, overpaddedDim,
                                      problemDim);
    }
    // dim % wg == 0.
    assertDivisible(
        builder, loc, problemDim, wg,
        llvm::join_items("", problemName, " must be divisible by ", wgName));
    // 1 <= wg <= dim
    assertBounds(builder, loc, wg, wgName, mkIntConst(builder, loc, 1), "1",
                 problemDim, problemName);
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
  assertCmp(builder, loc, smt::IntPredicate::le, wgMProd, maxVGPRsVal,
            "prod(wg_m) <= 512 (max_vgprs)");
  assertCmp(builder, loc, smt::IntPredicate::le, wgNProd, maxVGPRsVal,
            "prod(wg_n) <= 512 (max_vgprs)");

  // prod(wg) == prod(sg) * sg_cnt * mma
  assertDivisible(builder, loc, wgMProd, sgMDenom,
                  "prod(wg_m) must be divisible by prod(sg_m) * mma_m");
  assertDivisible(builder, loc, wgNProd, sgNDenom,
                  "prod(wg_n) must be divisible by prod(sg_n) * mma_n");
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
  // red_k >= 1
  assertCmp(builder, loc, smt::IntPredicate::ge, redK,
            mkIntConst(builder, loc, 1),
            llvm::join_items("", redKInnerName, " >= 1"));
  // problemInnerKAligned % (packed_k) == 0
  assertDivisible(builder, loc, problemInnerKAligned, packedK,
                  llvm::join_items("", "aligned inner k dim ", kInnerDimName,
                                   " must be divisible by ", redKInnerName,
                                   " * mma_k"));
  // packed_k <= max_vgprs
  assertCmp(builder, loc, smt::IntPredicate::le, packedK, maxVGPRsVal,
            llvm::join_items("", redKInnerName, " * mma_k <= 512 (max_vgprs)"));
  // packed_k <= aligned_dim_k
  assertCmp(builder, loc, smt::IntPredicate::le, packedK, problemInnerKAligned,
            llvm::join_items("", redKInnerName, " * mma_k <= aligned k dim"));

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
  // (lhs_bytes * wg_m * packed_k) + (rhs_bytes * wg_n * packed_k)
  auto [lhsBytesVal, rhsBytesVal] =
      getLhsRhsOperandBytes(builder, loc, linalgOp);
  Value lhsMem = smt::IntMulOp::create(
      builder, loc, ValueRange{lhsBytesVal, wgMProd, packedK});
  Value rhsMem = smt::IntMulOp::create(
      builder, loc, ValueRange{rhsBytesVal, wgNProd, packedK});
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

/// Forward-decl: attention emitter (defined further below -- has its own
/// knob template and constraint body distinct from contraction/conv).
static LogicalResult
emitVectorDistributeAttentionConstraintsForOp(Operation *rootOp,
                                              RootOpAttr rootOpAttr);

/// Emit constraints for a single root op under a supported LLVMGPU pipeline.
/// Supports linalg contraction/convolution under both VectorDistribute and
/// TileAndFuse, plus `iree_linalg_ext.online_attention` under VectorDistribute
/// (dispatched to its own emitter -- attention has a distinct knob template
/// and constraint body).
static LogicalResult
emitConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr,
                     IREE::GPU::LoweringPipeline pipeline) {
  // Attention path: route OnlineAttentionOp under VectorDistribute to the
  // attention emitter. Attention is not (yet) supported under TileAndFuse;
  // returning success() silently here matches the existing
  // "shape/op unsupported -> fall through to default heuristics" pattern
  // for the contraction-only entries below.
  if (isa<IREE::LinalgExt::OnlineAttentionOp>(rootOp)) {
    if (pipeline == IREE::GPU::LoweringPipeline::VectorDistribute) {
      return emitVectorDistributeAttentionConstraintsForOp(rootOp, rootOpAttr);
    }
    return success();
  }

  // Gate on contraction-like linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);

  bool isConv = linalg::isaConvolutionOpInterface(linalgOp);
  if (!linalgOp || (!linalg::isaContractionOpInterface(linalgOp) && !isConv)) {
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
  SmallVector<unsigned> fusedIgemmKDims = {};
  DictionaryAttr knobs;
  switch (pipeline) {
  case IREE::GPU::LoweringPipeline::VectorDistribute:
    knobs =
        buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
    break;
  case IREE::GPU::LoweringPipeline::TileAndFuse:
    if (isConv) {
      fusedIgemmKDims = getFusedIgemmKDimIndices(linalgOp, dims->k);
    }
    knobs = buildTileAndFuseKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs,
                                      /*isConv=*/isConv,
                                      /*fusedIgemmKDims=*/fusedIgemmKDims);
    break;
  default:
    return success();
  }
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(ctx, pipeline);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  switch (pipeline) {
  case IREE::GPU::LoweringPipeline::VectorDistribute:
    return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                           shell.smtDimArgs, compatibleMMAs);
  case IREE::GPU::LoweringPipeline::TileAndFuse:
    return emitTileAndFuseConstraints(
        builder, linalgOp, *dims, gpuTarget, shell.smtDimArgs, compatibleMMAs,
        {/*useIgemm=*/isConv, /*fusedIgemmKDims=*/fusedIgemmKDims});
  default:
    return success();
  }
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
    if (isa<IREE::LinalgExt::OnlineAttentionOp>(rootOp)) {
      return rootOp;
    }
  }
  return nullptr;
}

//===-------------------------------------------------------------------===//
// VectorDistribute attention constraints.
//
// Ports `generate_attention_vector_distribute_constraints` from amdsharktuner
// (rocm_dispatch_constraints.py:465-638) to SMT emission in IREE. This covers:
//   - QK and PV intrinsic divisibility against the attention dims
//   - Tile-size divisibility by subgroup counts and selected intrinsic sizes
//   - Subgroup-count bounds and subgroup-product pin
//   - Workgroup thread-count limit
//   - qk_acc == pv_acc layout matching
//   - QK-output reuse predicate derived from layout matches
//   - is_valid_vector_distribute_mma_schedule helpers for QK and PV
//   - Shared-memory budget, including the Python tuner's QK-output reuse term
//===-------------------------------------------------------------------===//

namespace {

/// Per-op decomposition + static sizes for an OnlineAttentionOp.
struct AttentionDimInfo {
  // Loop-dim index lists (positions into the iteration domain).
  SmallVector<int64_t> batch;
  SmallVector<int64_t> m;
  SmallVector<int64_t> k1;
  SmallVector<int64_t> k2;
  SmallVector<int64_t> n;
  unsigned domainRank;
  // Static sizes for the inner-most dim of each list. v0 only models the
  // innermost dim of each role; multi-dim per role is deferred.
  int64_t mSize;
  int64_t k1Size;
  int64_t k2Size;
  int64_t nSize;
  // Element types.
  Type queryElemType;
  Type keyElemType;
  Type valueElemType;
  Type outputElemType;
  // Transpose flags: true if the inner-most loop dim that the schedule
  // groups along is NOT in the last result position of the operand's
  // indexing map. The Python tuner's
  // `is_valid_vector_distribute_mma_schedule` uses these to choose
  // between m_wg_size and k_wg_size for the inner LHS/RHS dim.
  //   transposedQ: Q's k1 dim is not in the last result position.
  //   transposedK: K's k1 dim is not in the last result position.
  //   transposedV: V's k2 dim is not in the last result position.
  bool transposedQ;
  bool transposedK;
  bool transposedV;
};

} // namespace

/// Pull the attention-domain dim decomposition + static sizes out of an
/// `OnlineAttentionOp`. Returns `failure` if dim inference fails or any
/// of the inner-most M/K1/K2/N dims is dynamic (v0 limitation).
static FailureOr<AttentionDimInfo>
inferAttentionDimInfo(IREE::LinalgExt::OnlineAttentionOp attnOp) {
  auto detail = IREE::LinalgExt::AttentionOpDetail::get(
      attnOp.getQueryMap(), attnOp.getKeyMap(), attnOp.getValueMap(),
      attnOp.getOutputMap());
  if (failed(detail)) {
    return failure();
  }
  AttentionDimInfo info;
  info.batch = llvm::to_vector(detail->getBatchDims());
  info.m = llvm::to_vector(detail->getMDims());
  info.k1 = llvm::to_vector(detail->getK1Dims());
  info.k2 = llvm::to_vector(detail->getK2Dims());
  info.n = llvm::to_vector(detail->getNDims());
  info.domainRank = detail->getDomainRank();
  if (info.m.empty() || info.k1.empty() || info.k2.empty() || info.n.empty()) {
    return failure();
  }
  // v0 limitation: the constraint body uses a single SMT variable per
  // attention role (m_tile, n_tile, red_k2, qk_mma_k etc.) and indexes
  // `staticBounds` at the innermost dim of each role. If a role
  // decomposes into multiple iteration dims (e.g. multi-head attention
  // with M = [b_head, m_inner]), silently dropping the outer dims would
  // under-constrain the problem and emit garbage candidates. Reject the
  // multi-dim case explicitly until the constraint body learns to
  // factor across multiple dims per role.
  if (info.m.size() != 1 || info.k1.size() != 1 || info.k2.size() != 1 ||
      info.n.size() != 1) {
    return failure();
  }
  SmallVector<int64_t, 4> staticBounds = attnOp.getStaticLoopRanges();
  auto staticSize = [&](int64_t dim) -> int64_t {
    return (dim >= 0 && dim < (int64_t)staticBounds.size())
               ? staticBounds[dim]
               : ShapedType::kDynamic;
  };
  info.mSize = staticSize(info.m.back());
  info.k1Size = staticSize(info.k1.back());
  info.k2Size = staticSize(info.k2.back());
  info.nSize = staticSize(info.n.back());
  if (ShapedType::isDynamic(info.mSize) || ShapedType::isDynamic(info.k1Size) ||
      ShapedType::isDynamic(info.k2Size) || ShapedType::isDynamic(info.nSize)) {
    return failure();
  }
  info.queryElemType = getElementTypeOrSelf(attnOp.getQuery().getType());
  info.keyElemType = getElementTypeOrSelf(attnOp.getKey().getType());
  info.valueElemType = getElementTypeOrSelf(attnOp.getValue().getType());
  info.outputElemType = getElementTypeOrSelf(attnOp.getOutput().getType());

  // Compute transpose flags by checking whether the matmul's "natural
  // inner dim" sits in the last result position of each operand's
  // indexing map. Direct port of the Python tuner's per-operand flag
  // (sharktuner/constraint_generator.py:678-680):
  //   transposed_q = k1Dim != position_of_last_q_result   (Q's K dim is K1)
  //   transposed_k = k1Dim != position_of_last_k_result   (K's K dim is K1)
  //   transposed_v = k2Dim != position_of_last_v_result   (V's K dim is K2)
  // Note V uses K2 (PV matmul's contracting dim) not N -- keeping the
  // per-operand contracting-dim convention regardless of whether the
  // operand is the matmul's LHS or RHS. Schedule-validity inverts these
  // when picking the inner load dim (kWg vs nWg) for vector loads.
  auto innerDimIsLast = [&](AffineMap map, int64_t reductionDim) -> bool {
    if (map.getNumResults() == 0) {
      return false;
    }
    auto dimExpr =
        dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1));
    if (!dimExpr) {
      return false;
    }
    return static_cast<int64_t>(dimExpr.getPosition()) == reductionDim;
  };
  info.transposedQ = !innerDimIsLast(attnOp.getQueryMap(), info.k1.back());
  info.transposedK = !innerDimIsLast(attnOp.getKeyMap(), info.k1.back());
  info.transposedV = !innerDimIsLast(attnOp.getValueMap(), info.k2.back());
  return info;
}

/// Filter MMA intrinsics that can target a `(lhs, rhs, init) -> result`
/// matmul-shaped problem with the given element types and sizes. Used
/// twice for attention: once for QK (Q x K^T -> intermediate f32) and
/// once for PV (intermediate x V -> output). Mirrors `getCompatibleMMAAttrs`
/// but takes element types explicitly instead of reading them from a
/// `linalg::LinalgOp` (attention's operand structure is different).
static SmallVector<Attribute>
getCompatibleAttentionMMAAttrs(IREE::GPU::TargetAttr gpuTarget, int64_t mSize,
                               int64_t nSize, int64_t kSize, Type lhsElemType,
                               Type rhsElemType, Type initElemType) {
  if (gpuTarget.getWgp().getMma().empty()) {
    return {};
  }
  SmallVector<Attribute> mmaAttrs;
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();
  GPUMatmulShapeType problem{mSize,       nSize,       kSize,
                             lhsElemType, rhsElemType, initElemType};
  auto getIntrinsic = [](IREE::GPU::MMAAttr mma) -> GPUIntrinsicType {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  };
  auto getVirtualIntrinsic =
      [](IREE::GPU::VirtualMMAAttr vmma) -> GPUIntrinsicType {
    auto [m, n, k] = vmma.getMNKShape();
    auto [a, b, c] = vmma.getABCElementTypes();
    return GPUIntrinsicType{m, n, k, a, b, c, vmma};
  };

  // Square-mn filter (matches Python tuner): the attention constraint
  // formulation uses a single `intrinsic_mn` z3 variable for both M and N
  // intrinsic dims. That implicitly excludes MMAs with M != N -- e.g.
  // VDMFMA_F32_8x16x64x2_F16 (m=8, n=16). Mirroring the filter here keeps
  // the candidate sets aligned for parity. (Square-mn MMAs are the only
  // ones the IREE attention codegen currently exercises in practice.)
  auto isSquareMN = [](IREE::Codegen::InnerTileDescAttrInterface attr) -> bool {
    if (auto mma = dyn_cast<IREE::GPU::MMAAttr>(attr)) {
      auto [m, n, k] = mma.getMNKShape();
      return m == n;
    }
    if (auto vmma = dyn_cast<IREE::GPU::VirtualMMAAttr>(attr)) {
      auto [m, n, k] = vmma.getMNKShape();
      return m == n;
    }
    return false;
  };

  MLIRContext *ctx = gpuTarget.getContext();
  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    if (mma.isBlockIntrinsic()) {
      continue;
    }
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    if (!isSquareMN(mma)) {
      continue;
    }
    if (succeeded(canTargetIntrinsic(problem, getIntrinsic(mma),
                                     targetSubgroupSize, /*canUpcastAcc=*/true,
                                     /*mustBeAligned=*/false))) {
      if (!llvm::is_contained(mmaAttrs, mma)) {
        mmaAttrs.push_back(mma);
      }
    }
    // Virtual MMA variants of this MMA -- VMFMA / VDMFMA / etc.
    // Attention enumerates these (Python tuner sets
    // allow_virtual_mma=True). Each virtual variant is a different
    // tiling/unroll of the underlying MMA shape, so they have distinct layouts
    // that the match_layout constraints pick up.
    for (IREE::GPU::VirtualMMAIntrinsic vIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto vmma = IREE::GPU::VirtualMMAAttr::get(ctx, vIntrinsic);
      if (vmma.getSubgroupSize() != targetSubgroupSize) {
        continue;
      }
      if (!vmma.getDistributionMappingKind()) {
        continue;
      }
      if (!isSquareMN(vmma)) {
        continue;
      }
      if (failed(canTargetIntrinsic(problem, getVirtualIntrinsic(vmma),
                                    targetSubgroupSize,
                                    /*canUpcastAcc=*/true,
                                    /*mustBeAligned=*/false))) {
        continue;
      }
      if (!llvm::is_contained(mmaAttrs, Attribute(vmma))) {
        mmaAttrs.push_back(vmma);
      }
    }
  }
  return mmaAttrs;
}

/// Build a per-matmul `iree_gpu.lowering_config` knob template containing
/// every field IREE's attention codegen (`KernelConfig.cpp::setAttention...`)
/// expects: `mma_kind` (knob), `promote_operands` + matching
/// `promotion_types`, and `subgroup_basis` (the basis counts re-use the
/// top-level sg_m_cnt / sg_n_cnt knobs so the materialized per-matmul
/// counts stay consistent with the top-level subgroup assignment).
///
/// Must produce a `LoweringConfigAttr` (typed), not a raw `DictionaryAttr`,
/// because the attention decomposition (AggregatedOpInterfaceImpl) casts
/// `decomposition_config.qk_attrs.lowering_config` to `LoweringConfigAttr`
/// -- a raw dict here causes the cast to silently fail and codegen falls
/// back to default per-matmul behavior (every tuning candidate compiles
/// to the same binary).
///
/// `promotedOperands` is fixed per matmul: `{0, 1}` for QK (promote Q
/// and K), `{1}` for PV (promote V only); these mirror the native
/// codegen path's `appendPromotedOperandsListWithDerivedThread` calls.
///
/// `projectedDims` are the iteration-domain dims this matmul does NOT
/// touch (so they are projected out of the subgroup_basis mapping): N
/// dims for QK, K1 dims for PV.
static IREE::GPU::LoweringConfigAttr buildPerMatmulLoweringConfigAttr(
    MLIRContext *ctx, const AttentionDimInfo &dims, StringRef mmaIdxName,
    ArrayRef<Attribute> compatibleMMAs, ArrayRef<int64_t> promotedOperands,
    ArrayRef<int64_t> projectedDims) {
  SmallVector<NamedAttribute> entries;
  entries.emplace_back(kKnobMmaKindKey,
                       OneOfKnobAttr::get(ctx, mmaIdxName, compatibleMMAs));

  SmallVector<Attribute> promoteEntries;
  promoteEntries.reserve(promotedOperands.size());
  for (int64_t idx : promotedOperands) {
    promoteEntries.push_back(makeIntAttr(ctx, idx));
  }
  entries.emplace_back(kKnobPromoteOperandsKey,
                       ArrayAttr::get(ctx, promoteEntries));

  SmallVector<Attribute> promotionTypes(
      promotedOperands.size(), IREE::GPU::DerivedThreadConfigAttr::get(ctx));
  entries.emplace_back(StringRef("promotion_types"),
                       ArrayAttr::get(ctx, promotionTypes));

  // subgroup_basis = [[counts], [mapping]] where counts has one entry per
  // iteration-domain dim (with sg_m_cnt / sg_n_cnt knobs at the M / N
  // positions, 1 elsewhere) and mapping is 0..domainRank-1 minus the
  // projected dims.
  SmallVector<Attribute> counts(dims.domainRank, makeIntAttr(ctx, 1));
  counts[dims.m.back()] = IntKnobAttr::get(ctx, kKnobSgMCntName);
  counts[dims.n.back()] = IntKnobAttr::get(ctx, kKnobSgNCntName);
  llvm::SmallDenseSet<int64_t, 4> projectedSet(projectedDims.begin(),
                                               projectedDims.end());
  SmallVector<Attribute> mapping;
  for (int64_t i = 0, e = static_cast<int64_t>(dims.domainRank); i < e; ++i) {
    if (!projectedSet.contains(i)) {
      mapping.push_back(makeIntAttr(ctx, i));
    }
  }
  SmallVector<Attribute> basisPair = {ArrayAttr::get(ctx, counts),
                                      ArrayAttr::get(ctx, mapping)};
  entries.emplace_back(StringRef("subgroup_basis"),
                       ArrayAttr::get(ctx, basisPair));

  return IREE::GPU::LoweringConfigAttr::get(ctx,
                                            DictionaryAttr::get(ctx, entries));
}

/// SSA values for the 6 per-(matmul, operand) layout fields, returned from
/// `emitLayoutValues` so downstream constraints (match_layout in v1.4, the
/// qk_acc == pv_acc invariant below) can reuse them.
struct LayoutValues {
  Value elementX;
  Value elementY;
  Value threadX;
  Value threadY;
  Value tstridesX;
  Value tstridesY;
};

/// Create the 6 derived layout values for a single (matmul, operand) pair via
/// `iree_codegen.smt.lookup` ops over the MMA index knob. Returns the SSA
/// lookup results so the caller can compose match_layout-style assertions on
/// them.
///
/// `mmaIdxValue` is the *already-created* SSA value for the MMA index
/// knob (the qk_mma_idx / pv_mma_idx knob constructed by `emitMMALookup` for
/// the shape lookups).
///
/// `operandIndex` is one of `IREE::GPU::kMMAOperand{Lhs,Rhs,Acc}`. The
/// `MMASingleSubgroupLayout` returned by `getSingleSubgroupLayout` has
/// 2-element vectors for `element`, `thread`, `tstrides` (indexed [0]=x,
/// [1]=y); we only emit the 6 fields the v1 `matchLayout` test compares (outer
/// is omitted -- no constraint depends on it).
static LayoutValues emitLayoutValues(OpBuilder &builder, Location loc,
                                     ArrayRef<Attribute> mmaList,
                                     Value mmaIdxValue, int operandIndex) {
  LayoutValues result{};
  if (mmaList.empty()) {
    return result;
  }

  SmallVector<int64_t> keys;
  SmallVector<int64_t> elementX, elementY;
  SmallVector<int64_t> threadX, threadY;
  SmallVector<int64_t> tstridesX, tstridesY;
  for (auto [i, attr] : llvm::enumerate(mmaList)) {
    auto interfaceAttr = cast<IREE::Codegen::InnerTileDescAttrInterface>(attr);
    IREE::GPU::MMASingleSubgroupLayout layout =
        IREE::GPU::getSingleSubgroupLayout(interfaceAttr, operandIndex);
    // Each of element/thread/tstrides is a 2-element vector in canonical
    // (x, y) order. If anything is degenerate (size < 2), treat the
    // missing entry as 1 so the lookup table is well-formed.
    auto safeGet = [](ArrayRef<int64_t> v, size_t k) -> int64_t {
      return k < v.size() ? v[k] : 1;
    };
    keys.push_back(static_cast<int64_t>(i));
    elementX.push_back(safeGet(layout.element, 0));
    elementY.push_back(safeGet(layout.element, 1));
    threadX.push_back(safeGet(layout.thread, 0));
    threadY.push_back(safeGet(layout.thread, 1));
    tstridesX.push_back(safeGet(layout.tstrides, 0));
    tstridesY.push_back(safeGet(layout.tstrides, 1));
  }

  auto intTy = smt::IntType::get(builder.getContext());
  auto lookupField = [&](ArrayRef<int64_t> values) -> Value {
    return LookupOp::create(builder, loc, intTy, mmaIdxValue, keys, values);
  };
  result.elementX = lookupField(elementX);
  result.elementY = lookupField(elementY);
  result.threadX = lookupField(threadX);
  result.threadY = lookupField(threadY);
  result.tstridesX = lookupField(tstridesX);
  result.tstridesY = lookupField(tstridesY);
  return result;
}

/// Emit the v1 schedule-validity constraints
/// (`is_valid_vector_distribute_mma_schedule` from the Python tuner).
/// For a single matmul schedule, asserts:
///   1. `schedule_aligned`: matmul shape divides the derived workgroup tile on
///      each of M, N, K -- i.e. the workgroup tile partitions the matmul
///      cleanly.
///   2. `lhs_distributable` / `rhs_distributable`: the inner load/store
///      dimension of each operand divides either
///      (wg_threads x elems_per_thread) or vice versa -- so vectorized
///      loads at the max-vector-load bitwidth distribute across the
///      workgroup without remainder.
///
/// `transposedLhs` / `transposedRhs` flip which axis of the workgroup
/// tile is the "inner" dim (the contiguous one for vector loads). For
/// LHS: K is inner when not transposed, M is inner when transposed.
/// For RHS: N is inner when not transposed, K is inner when transposed.
///
/// `rhsBitwidth` is the bit width of the RHS element type -- drives the
/// elements-per-thread count for the inner-dim vectorization check.
/// (The Python tuner uses RHS type for both LHS and RHS; mirroring.)
static void emitScheduleValidityConstraints(
    OpBuilder &builder, Location loc, StringRef matmulName, Value matmulM,
    Value matmulN, Value matmulK, Value mSize, Value nSize, Value kSize,
    Value mSubgroupCount, Value nSubgroupCount, Value mTileCount,
    Value nTileCount, Value kTileCount, Value wgThreads, bool transposedLhs,
    bool transposedRhs, unsigned rhsBitwidth) {
  Value mPartition = smt::IntMulOp::create(
      builder, loc, ValueRange{mSubgroupCount, mTileCount, mSize});
  Value nPartition = smt::IntMulOp::create(
      builder, loc, ValueRange{nSubgroupCount, nTileCount, nSize});
  Value kPartition =
      smt::IntMulOp::create(builder, loc, ValueRange{kTileCount, kSize});
  assertDivisible(builder, loc, matmulM, mPartition,
                  llvm::formatv("{0}.m must be divisible by the derived M "
                                "workgroup tile",
                                matmulName)
                      .str());
  assertDivisible(builder, loc, matmulN, nPartition,
                  llvm::formatv("{0}.n must be divisible by the derived N "
                                "workgroup tile",
                                matmulName)
                      .str());
  assertDivisible(builder, loc, matmulK, kPartition,
                  llvm::formatv("{0}.k must be divisible by the derived K "
                                "workgroup tile",
                                matmulName)
                      .str());

  // lhs/rhs distributable: inner_dim / elems_per_thread distributes
  // across wg_threads. The Python tuner picks elems_per_thread =
  // max_vector_load_bitwidth (128) / rhs_bitwidth. 128 is the AMD
  // global_load_dwordx4 / ds_read_b128 width on gfx9 and the
  // `max_load_instruction_bits` value all GPU targets in
  // `gpu_targets.json` advertise -- the constant should track that
  // descriptor field if it ever varies per arch.
  constexpr int64_t kMaxVectorLoadBitWidth = 128;
  if (rhsBitwidth < 8) {
    // Sub-byte or unknown bitwidth -- skip vector-load distribution
    // check (matches the Python tuner's silent fallthrough for
    // dynamic shapes; revisit when sub-byte support lands).
    return;
  }
  int64_t elemsPerThread = kMaxVectorLoadBitWidth / rhsBitwidth;
  if (elemsPerThread <= 0) {
    return;
  }
  Value elemsPerThreadVal = mkIntConst(builder, loc, elemsPerThread);

  // Mirrors GPUMMASchedule.m/n/k_wg_size in rocm_dispatch_constraints.py.
  Value mWg = mPartition;
  Value nWg = nPartition;
  Value kWg = kPartition;

  Value innerLhs = transposedLhs ? mWg : kWg;
  Value innerRhs = transposedRhs ? kWg : nWg;
  // NOTE: Tightening zero-division schedule validity is tracked with the
  // Python-parity follow-ups in TODO(#23535) near the attention emitter.
  Value lhsDiv =
      smt::IntDivOp::create(builder, loc, innerLhs, elemsPerThreadVal);
  Value rhsDiv =
      smt::IntDivOp::create(builder, loc, innerRhs, elemsPerThreadVal);

  // distributable: `inner_div % wg_threads == 0  OR  wg_threads %
  // inner_div == 0`. Encoded as the disjunction of two `eq 0` ops.
  auto mkDistributable = [&](Value innerDiv, StringRef operandName) {
    Value zero = mkIntConst(builder, loc, 0);
    Value mod1 = smt::IntModOp::create(builder, loc, innerDiv, wgThreads);
    Value cmp1 = smt::EqOp::create(builder, loc, mod1, zero);
    Value mod2 = smt::IntModOp::create(builder, loc, wgThreads, innerDiv);
    Value cmp2 = smt::EqOp::create(builder, loc, mod2, zero);
    Value either = smt::OrOp::create(builder, loc, ValueRange{cmp1, cmp2});
    AssertOp::create(
        builder, loc, either,
        llvm::formatv("{0}.{1} inner dim distributable across wg_threads",
                      matmulName, operandName)
            .str());
  };
  mkDistributable(lhsDiv, "lhs");
  mkDistributable(rhsDiv, "rhs");
}

/// Build the attention knob dictionary. Mirrors the top-level attention
/// lowering_config (`workgroup`, `reduction`, `promote_operands`)
/// plus the nested `decomposition_config` carrying per-matmul
/// `lowering_config`s for QK and PV.
static DictionaryAttr
buildVectorDistributeAttentionKnobsDict(MLIRContext *ctx,
                                        const AttentionDimInfo &dims,
                                        ArrayRef<Attribute> compatibleQKMMAs,
                                        ArrayRef<Attribute> compatiblePVMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // workgroup: batch=1, M dim gets m_tile knob, K1/K2=0, N dim gets n_tile
  // knob. Batch entries set to 1 (each workgroup handles one batch slice) --
  // matches native attention codegen output (KernelConfig.cpp's
  // setAttention...).
  SmallVector<Attribute> workgroupEntries(dims.domainRank,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (int64_t b : dims.batch) {
    workgroupEntries[b] = makeIntAttr(ctx, 1);
  }
  workgroupEntries[dims.m.back()] = IntKnobAttr::get(ctx, kKnobAttnMTileName);
  workgroupEntries[dims.n.back()] = IntKnobAttr::get(ctx, kKnobAttnNTileName);
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));

  // reduction: K2 dim gets red_k2 knob, others 0. Attention codegen reads
  // this under the key `reduction` (not `partial_reduction`, which is a
  // different tiling level -- see GPULoweringConfigUtils.cpp's level names).
  SmallVector<Attribute> reductionEntries(dims.domainRank,
                                          makeIntAttr(ctx, kNoTileDimVal));
  reductionEntries[dims.k2.back()] = IntKnobAttr::get(ctx, kKnobAttnK2TileName);
  knobsEntries.emplace_back(StringRef("reduction"),
                            ArrayAttr::get(ctx, reductionEntries));

  // promote_operands = [0, 1, 2] (Q, K, V; fixed for v0) with matching
  // promotion_types = [derived_thread_config, derived_thread_config,
  //                    derived_thread_config].
  SmallVector<Attribute> promoteEntries = {
      makeIntAttr(ctx, 0), makeIntAttr(ctx, 1), makeIntAttr(ctx, 2)};
  knobsEntries.emplace_back(kKnobPromoteOperandsKey,
                            ArrayAttr::get(ctx, promoteEntries));
  SmallVector<Attribute> promotionTypes(
      3, IREE::GPU::DerivedThreadConfigAttr::get(ctx));
  knobsEntries.emplace_back(StringRef("promotion_types"),
                            ArrayAttr::get(ctx, promotionTypes));

  // decomposition_config = {qk_attrs = {lowering_config = ...},
  //                          pv_attrs = {lowering_config = ...}}.
  // Each inner lowering_config is a typed LoweringConfigAttr carrying
  // mma_kind + promote_operands + promotion_types + subgroup_basis,
  // because IREE attention decomposition casts the inner attribute to
  // LoweringConfigAttr (a raw DictAttr would be silently rejected).
  IREE::GPU::LoweringConfigAttr qkLoweringConfig =
      buildPerMatmulLoweringConfigAttr(
          ctx, dims, kKnobQKMmaIdxName, compatibleQKMMAs,
          /*promotedOperands=*/{0, 1}, /*projectedDims=*/dims.n);
  IREE::GPU::LoweringConfigAttr pvLoweringConfig =
      buildPerMatmulLoweringConfigAttr(
          ctx, dims, kKnobPVMmaIdxName, compatiblePVMMAs,
          /*promotedOperands=*/{1}, /*projectedDims=*/dims.k1);
  SmallVector<NamedAttribute> qkAttrs = {
      {StringAttr::get(ctx, kKnobLoweringConfigKey), qkLoweringConfig}};
  SmallVector<NamedAttribute> pvAttrs = {
      {StringAttr::get(ctx, kKnobLoweringConfigKey), pvLoweringConfig}};
  SmallVector<NamedAttribute> decompEntries = {
      {StringAttr::get(ctx, kKnobQKAttrsKey),
       DictionaryAttr::get(ctx, qkAttrs)},
      {StringAttr::get(ctx, kKnobPVAttrsKey),
       DictionaryAttr::get(ctx, pvAttrs)},
  };
  knobsEntries.emplace_back(kDecompositionConfigKey,
                            DictionaryAttr::get(ctx, decompEntries));
  // Note: top-level no longer carries `subgroup_m_count` /
  // `subgroup_n_count` -- the per-matmul `subgroup_basis` above is what
  // attention codegen consumes for subgroup partitioning. The sg_m_cnt /
  // sg_n_cnt knobs are still declared (and constrained) via the per-matmul
  // basis counts, so the materializer wires them up consistently.

  // workgroup_size: 3D, x = derived from sg_cnts * sg_size, y/z fixed at 1.
  SmallVector<Attribute> wgSizeKnobs = {IntKnobAttr::get(ctx, kKnobWgSizeXName),
                                        makeIntAttr(ctx, 1),
                                        makeIntAttr(ctx, 1)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit the attention constraint body. See the section header above for the
/// Python tuner pieces covered here.
static LogicalResult emitVectorDistributeAttentionConstraints(
    OpBuilder &builder, IREE::LinalgExt::OnlineAttentionOp attnOp,
    const AttentionDimInfo &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleQKMMAs,
    ArrayRef<Attribute> compatiblePVMMAs) {
  Location loc = attnOp.getLoc();

  // Hardware constants.
  Value subgroupSizeVal =
      mkIntConst(builder, loc, gpuTarget.getPreferredSubgroupSize());
  Value maxThreadsVal = mkIntConst(
      builder, loc, gpuTarget.getWgp().getMaxThreadCountPerWorkgroup());
  Value maxSharedMemVal =
      mkIntConst(builder, loc, gpuTarget.getWgp().getMaxWorkgroupMemoryBytes());
  // VGPR-budgeted upper bound on any single tile axis (mirrors the matmul
  // emitter's 512 constant and the Python tuner's
  // `rocm_dispatch_constraints.py:generate_attention_vector_distribute_constraints`
  // `max_tile = 512` line). 512 is the per-thread VGPR count on CDNA3
  // (gfx942); revisit when the target descriptor exposes a per-thread
  // VGPR count instead of hardcoding it here.
  Value maxVGPRsVal = mkIntConst(builder, loc, 512);
  Value oneVal = mkIntConst(builder, loc, 1);

  // Per-dim SMT arg references.
  unsigned mDim = dims.m.back();
  unsigned nDim = dims.n.back();
  unsigned k2Dim = dims.k2.back();
  std::string mDimName = makeVarName(kLoopRangePrefix, mDim);
  std::string nDimName = makeVarName(kLoopRangePrefix, nDim);
  std::string k2DimName = makeVarName(kLoopRangePrefix, k2Dim);

  // Knobs.
  Value mTile = mkKnob(builder, loc, kKnobAttnMTileName);
  Value nTile = mkKnob(builder, loc, kKnobAttnNTileName);
  Value k2Tile = mkKnob(builder, loc, kKnobAttnK2TileName);
  Value sgMCnt = mkKnob(builder, loc, kKnobSgMCntName);
  Value sgNCnt = mkKnob(builder, loc, kKnobSgNCntName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);

  // Per-matmul intrinsic shape lookups. Each yields (mma_m, mma_n, mma_k).
  // For attention: QK shapes correspond to the QxK^T matmul, PV shapes
  // to the PxV matmul (different K dimensions). `emitMMALookup` creates
  // the MMA-index knob with the name passed in, so the two attention
  // matmuls get distinct `qk_mma_idx` / `pv_mma_idx` knobs.
  auto [qkMmaIdx, qkMmaM, qkMmaN, qkMmaK] =
      emitMMALookup(builder, loc, compatibleQKMMAs, kKnobQKMmaIdxName);
  auto [pvMmaIdx, pvMmaM, pvMmaN, pvMmaK] =
      emitMMALookup(builder, loc, compatiblePVMMAs, kKnobPVMmaIdxName);

  // Single-intrinsic-mn invariant. The Python tuner uses ONE
  // `intrinsic_mn` SMT variable across both attention matmuls (see
  // rocm_dispatch_constraints.py:generate_attention_vector_distribute_constraints);
  // we model the two MMAs as independent `one_of_knob`s for binding
  // flexibility, then re-impose the equality here so the candidate set
  // matches. Without these asserts, a satisfying solution could pick
  // (qk=32x32x8, pv=16x16x16) -- the layout-match invariant rejects
  // most such cross combinations in practice (different MMA shapes
  // produce different per-subgroup layouts) but the explicit shape
  // equality is the load-bearing constraint, not an emergent property
  // of layout matching. Belt-and-suspenders.
  //
  // Placement: these must remain after both `emitMMALookup` calls above; the
  // asserts reference the `qkMmaM`/`pvMmaM`/`qkMmaN`/`pvMmaN` SSA values those
  // produce.
  assertEq(builder, loc, qkMmaM, pvMmaM,
           "qk_mma_m == pv_mma_m (single intrinsic_mn)");
  assertEq(builder, loc, qkMmaN, pvMmaN,
           "qk_mma_n == pv_mma_n (single intrinsic_mn)");

  // Derive per-operand layout fields (element/thread/tstrides x x/y) from the
  // chosen MMA via lookup ops. The layout-match asserts below (and the v1.4
  // match_layout-derived booleans) can use these SSA values directly; they do
  // not need separate solver knobs.
  LayoutValues qkAccLayout = emitLayoutValues(
      builder, loc, compatibleQKMMAs, qkMmaIdx, IREE::GPU::kMMAOperandAcc);
  LayoutValues pvLhsLayout = emitLayoutValues(
      builder, loc, compatiblePVMMAs, pvMmaIdx, IREE::GPU::kMMAOperandLhs);
  LayoutValues pvRhsLayout = emitLayoutValues(
      builder, loc, compatiblePVMMAs, pvMmaIdx, IREE::GPU::kMMAOperandRhs);
  LayoutValues pvAccLayout = emitLayoutValues(
      builder, loc, compatiblePVMMAs, pvMmaIdx, IREE::GPU::kMMAOperandAcc);

  // match_layout(a, b): conjunction of the 6 field equalities. Returns an
  // !smt.bool so callers can directly compose the Python tuner's
  // match-layout-derived predicates.
  auto buildMatchLayoutBool = [&](const LayoutValues &a,
                                  const LayoutValues &b) -> Value {
    Value eqEx = smt::EqOp::create(builder, loc, a.elementX, b.elementX);
    Value eqEy = smt::EqOp::create(builder, loc, a.elementY, b.elementY);
    Value eqTx = smt::EqOp::create(builder, loc, a.threadX, b.threadX);
    Value eqTy = smt::EqOp::create(builder, loc, a.threadY, b.threadY);
    Value eqSx = smt::EqOp::create(builder, loc, a.tstridesX, b.tstridesX);
    Value eqSy = smt::EqOp::create(builder, loc, a.tstridesY, b.tstridesY);
    return smt::AndOp::create(builder, loc,
                              ValueRange{eqEx, eqEy, eqTx, eqTy, eqSx, eqSy});
  };

  // Derived match_layout booleans. Matching QK acc with PV rhs controls
  // col-major materialization; matching either PV operand controls QK-output
  // reuse in the shared-memory budget below.
  Value qkAccMatchesPvRhs = buildMatchLayoutBool(qkAccLayout, pvRhsLayout);
  Value qkAccMatchesPvLhs = buildMatchLayoutBool(qkAccLayout, pvLhsLayout);
  Value canReuseIsTrue = smt::OrOp::create(
      builder, loc, ValueRange{qkAccMatchesPvLhs, qkAccMatchesPvRhs});

  // Layout-match invariant: QK accumulator layout MUST equal PV
  // accumulator layout. This is the structural correctness constraint
  // for online attention -- the running max/sum bookkeeping requires
  // QK's output and PV's output to live in the same per-subgroup
  // layout so they accumulate over the same partition.
  // (Python tuner: rocm_dispatch_constraints.py:527,
  //   `match_layout(qk_mma_acc_layout, pv_mma_acc_layout)`.)
  assertEq(builder, loc, qkAccLayout.elementX, pvAccLayout.elementX,
           "qk_acc.element_x == pv_acc.element_x");
  assertEq(builder, loc, qkAccLayout.elementY, pvAccLayout.elementY,
           "qk_acc.element_y == pv_acc.element_y");
  assertEq(builder, loc, qkAccLayout.threadX, pvAccLayout.threadX,
           "qk_acc.thread_x == pv_acc.thread_x");
  assertEq(builder, loc, qkAccLayout.threadY, pvAccLayout.threadY,
           "qk_acc.thread_y == pv_acc.thread_y");
  assertEq(builder, loc, qkAccLayout.tstridesX, pvAccLayout.tstridesX,
           "qk_acc.tstrides_x == pv_acc.tstrides_x");
  assertEq(builder, loc, qkAccLayout.tstridesY, pvAccLayout.tstridesY,
           "qk_acc.tstrides_y == pv_acc.tstrides_y");

  // Constraint 0: sg_size pinned to the target's preferred subgroup size.
  assertEq(builder, loc, sgSize, subgroupSizeVal,
           "sg_size == preferred_subgroup_size");

  // Constraint 1: tile sizes must divide problem sizes.
  assertDivisible(
      builder, loc, smtDimArgs[mDim], mTile,
      llvm::formatv("{0} must be divisible by m_tile", mDimName).str());
  assertDivisible(
      builder, loc, smtDimArgs[nDim], nTile,
      llvm::formatv("{0} must be divisible by n_tile", nDimName).str());
  assertDivisible(
      builder, loc, smtDimArgs[k2Dim], k2Tile,
      llvm::formatv("{0} must be divisible by red_k2", k2DimName).str());

  // Constraint 2: tile sizes bounded by pv intrinsic shape and VGPR cap.
  // (Python tuner uses PV intrinsic for the attention-domain tile bounds.)
  assertCmp(builder, loc, smt::IntPredicate::ge, mTile, pvMmaM,
            "m_tile >= pv_mma_m");
  assertCmp(builder, loc, smt::IntPredicate::ge, nTile, pvMmaN,
            "n_tile >= pv_mma_n");
  assertCmp(builder, loc, smt::IntPredicate::ge, k2Tile, pvMmaK,
            "red_k2 >= pv_mma_k");
  assertCmp(builder, loc, smt::IntPredicate::le, mTile, maxVGPRsVal,
            "m_tile <= 512 (max VGPRs)");
  assertCmp(builder, loc, smt::IntPredicate::le, nTile, maxVGPRsVal,
            "n_tile <= 512 (max VGPRs)");
  assertCmp(builder, loc, smt::IntPredicate::le, k2Tile, maxVGPRsVal,
            "red_k2 <= 512 (max VGPRs)");

  // Constraint 3: tile-size factorization. Derive schedule partition factors
  // from the materialized tile sizes, subgroup counts, and selected MMA
  // intrinsic sizes. The equality after each quotient keeps the division exact.
  Value mScheduleDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, pvMmaM});
  Value mScheduleQuotient =
      smt::IntDivOp::create(builder, loc, mTile, mScheduleDenom);
  Value mFactor = smt::IntMulOp::create(
      builder, loc, ValueRange{mScheduleDenom, mScheduleQuotient});
  assertEq(builder, loc, mTile, mFactor,
           "m_tile must be divisible by sg_m_cnt * pv_mma_m");
  Value nScheduleDenom =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, pvMmaN});
  Value nScheduleQuotient =
      smt::IntDivOp::create(builder, loc, nTile, nScheduleDenom);
  Value nFactor = smt::IntMulOp::create(
      builder, loc, ValueRange{nScheduleDenom, nScheduleQuotient});
  assertEq(builder, loc, nTile, nFactor,
           "n_tile must be divisible by sg_n_cnt * pv_mma_n");
  Value k2ScheduleQuotient =
      smt::IntDivOp::create(builder, loc, k2Tile, pvMmaK);
  Value k2Factor = smt::IntMulOp::create(
      builder, loc, ValueRange{pvMmaK, k2ScheduleQuotient});
  assertEq(builder, loc, k2Tile, k2Factor,
           "red_k2 must be divisible by pv_mma_k");

  // QK K1 dim must be a whole multiple of qk_mma_k (Q x K^T has its
  // own K dim, distinct from the attention-domain K2).
  Value k1Static = mkIntConst(builder, loc, dims.k1Size);
  assertDivisible(builder, loc, k1Static, qkMmaK,
                  "dim_k1 must be divisible by qk_mma_k");

  // Constraint 4: subgroup count bounds. Mirrors the Python tuner's
  // `rocm_dispatch_constraints.py:generate_attention_vector_distribute_constraints`
  // bounds: sg_m_cnt in [1, 32] (32 = max VGPRs / min MMA size = 512 / 16
  // for the smallest valid MFMA intrinsic) and sg_n_cnt pinned to 1
  // (attention's PV matmul partitions only along M for v0).
  Value maxSgCntVal = mkIntConst(builder, loc, 32);
  assertBounds(builder, loc, sgMCnt, "sg_m_cnt", oneVal, "1", maxSgCntVal,
               "32");
  assertEq(builder, loc, sgNCnt, oneVal, "sg_n_cnt == 1");

  // Constraint 5: subgroup-product pin. The Python tuner hard-codes
  // `num_subgroups = 4` (rocm_dispatch_constraints.py default) for the
  // attention path -- empirically the best LDS/occupancy trade-off across
  // the heads/seq-lens we tune over. Encoded as an equality so the
  // candidate set matches the legacy generator exactly.
  Value sgNum = smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt});
  Value numSubgroupsVal = mkIntConst(builder, loc, 4);
  assertEq(builder, loc, sgNum, numSubgroupsVal, "sg_m_cnt * sg_n_cnt == 4");

  // Constraint 6: total thread count fits per-workgroup limit.
  Value totalThreads =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNum, sgSize});
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreads, maxThreadsVal,
            "sg_m_cnt * sg_n_cnt * sg_size <= max_threads");

  // Constraint 7: wg_size_x derivation (1D workgroup; y/z fixed in template).
  assertEq(builder, loc, wgSizeX, totalThreads,
           "wg_size_x == sg_m_cnt * sg_n_cnt * sg_size");

  // Constraint 7b: schedule-validity for QK and PV matmuls (ports
  // `is_valid_vector_distribute_mma_schedule` from the Python tuner).
  // Note QK's N tile uses PV's K intrinsic dim; this is the intermediate dim
  // shared between the two matmuls. QK's K schedule quotient is derived from
  // the static K1 size and selected QK intrinsic K.
  unsigned rhsBitsQK = dims.keyElemType.getIntOrFloatBitWidth();
  unsigned rhsBitsPV = dims.valueElemType.getIntOrFloatBitWidth();

  // QK schedule: matmul is (M, K2, K1) with M=mSize, N=k2Size, K=k1Size.
  Value qkM = mkIntConst(builder, loc, dims.mSize);
  Value qkN = mkIntConst(builder, loc, dims.k2Size);
  Value qkK = k1Static;
  Value qkKScheduleQuotient =
      smt::IntDivOp::create(builder, loc, k1Static, qkMmaK);
  emitScheduleValidityConstraints(
      builder, loc, "qk_schedule", qkM, qkN, qkK,
      /*mSize=*/pvMmaM, /*nSize=*/pvMmaK, /*kSize=*/qkMmaK,
      /*mSubgroupCount=*/sgMCnt, /*nSubgroupCount=*/oneVal,
      /*mTileCount=*/mScheduleQuotient,
      /*nTileCount=*/k2ScheduleQuotient,
      /*kTileCount=*/qkKScheduleQuotient, /*wgThreads=*/totalThreads,
      dims.transposedQ, dims.transposedK, rhsBitsQK);

  // PV schedule: matmul is (M, N, K2) with M=mSize, N=nSize, K=k2Size.
  Value pvM = mkIntConst(builder, loc, dims.mSize);
  Value pvN = mkIntConst(builder, loc, dims.nSize);
  Value pvK = mkIntConst(builder, loc, dims.k2Size);
  emitScheduleValidityConstraints(
      builder, loc, "pv_schedule", pvM, pvN, pvK,
      /*mSize=*/pvMmaM, /*nSize=*/pvMmaN, /*kSize=*/pvMmaK,
      /*mSubgroupCount=*/sgMCnt, /*nSubgroupCount=*/sgNCnt,
      /*mTileCount=*/mScheduleQuotient,
      /*nTileCount=*/nScheduleQuotient,
      /*kTileCount=*/k2ScheduleQuotient, /*wgThreads=*/totalThreads,
      /*transposedLhs=*/false, dims.transposedV, rhsBitsPV);

  // Constraint 8: shared memory budget. Mirrors the Python tuner's
  // per-matmul tile-based formula (rocm_dispatch_constraints.py:444-461,
  // 618-630): each matmul's shared-memory usage is the LHS-tile bytes
  // plus the RHS-tile bytes. The total is
  //   qk_shared + (can_reuse ? pv_shared / 2 : pv_shared)
  // where reuse means PV's LHS shares memory with QK's accumulator (so PV LHS
  // doesn't need its own slot).
  //
  // Skipped entirely if any element type is sub-byte (v0 caveat,
  // matches matmul emitter behavior).
  auto wholeByteSize = [&](Type ty) -> int64_t {
    unsigned bitWidth = IREE::Util::getTypeBitWidth(ty);
    if (bitWidth < 8) {
      return -1;
    }
    return bitWidth / 8;
  };
  int64_t qBytes = wholeByteSize(dims.queryElemType);
  int64_t kBytes = wholeByteSize(dims.keyElemType);
  int64_t vBytes = wholeByteSize(dims.valueElemType);
  if (qBytes > 0 && kBytes > 0 && vBytes > 0) {
    // QK schedule tile dims (matches the qk_schedule struct used in
    // schedule-validity above):
    //   tile_m = m_tile
    //   tile_n = red_k2
    //   tile_k = mma_qk_k x (k1Static / mma_qk_k) == k1Static
    Value pvMmaMVal = pvMmaM;
    Value pvMmaKVal = pvMmaK;
    Value qkTileM = smt::IntMulOp::create(
        builder, loc, ValueRange{pvMmaMVal, mScheduleQuotient, sgMCnt});
    Value qkTileN = smt::IntMulOp::create(
        builder, loc, ValueRange{pvMmaKVal, k2ScheduleQuotient});
    Value qkTileK = k1Static; // qk_intrinsic_k x (k1 / qk_intrinsic_k) = k1
    Value qkLhsBytes = smt::IntMulOp::create(
        builder, loc,
        ValueRange{qkTileM, qkTileK, mkIntConst(builder, loc, qBytes)});
    Value qkRhsBytes = smt::IntMulOp::create(
        builder, loc,
        ValueRange{qkTileN, qkTileK, mkIntConst(builder, loc, kBytes)});
    Value qkShared =
        smt::IntAddOp::create(builder, loc, ValueRange{qkLhsBytes, qkRhsBytes});

    // PV schedule tile dims:
    //   tile_m = m_tile
    //   tile_n = n_tile
    //   tile_k = red_k2
    Value pvTileM = qkTileM;
    Value pvTileN = smt::IntMulOp::create(
        builder, loc, ValueRange{pvMmaN, nScheduleQuotient, sgNCnt});
    Value pvTileK = smt::IntMulOp::create(
        builder, loc, ValueRange{pvMmaKVal, k2ScheduleQuotient});
    // PV LHS/RHS dtype == value dtype, matching the Python dispatch parser's
    // PV matmul model.
    Value pvLhsBytes = smt::IntMulOp::create(
        builder, loc,
        ValueRange{pvTileM, pvTileK, mkIntConst(builder, loc, vBytes)});
    Value pvRhsBytes = smt::IntMulOp::create(
        builder, loc,
        ValueRange{pvTileN, pvTileK, mkIntConst(builder, loc, vBytes)});
    Value pvShared =
        smt::IntAddOp::create(builder, loc, ValueRange{pvLhsBytes, pvRhsBytes});

    // If QK output can be reused as PV input, PV's LHS slot is shared with QK's
    // accumulator and PV's LHS bytes drop out of the total. The Python tuner
    // approximates this by halving pv_shared
    // (rocm_dispatch_constraints.py:622 `pv_shared //= 2`), which is
    // exact when pvLhsBytes == pvRhsBytes (square tiles, equal
    // operand widths -- the common attention case) and slightly off
    // otherwise. Mirroring the approximation here keeps the candidate
    // set bijective vs the Python tuner; switching to the structurally
    // exact `qkShared + pvRhsBytes` would diverge.
    //
    // The division floors (off-by-one in the LDS-friendly direction --
    // it admits at most one extra byte's worth of candidates that the
    // codegen would still accept), matching Python's `//` semantics.
    Value pvSharedHalf = smt::IntDivOp::create(builder, loc, pvShared,
                                               mkIntConst(builder, loc, 2));
    Value totalShared = smt::IteOp::create(builder, loc, canReuseIsTrue,
                                           pvSharedHalf, pvShared);
    totalShared =
        smt::IntAddOp::create(builder, loc, ValueRange{qkShared, totalShared});
    assertCmp(builder, loc, smt::IntPredicate::le, totalShared, maxSharedMemVal,
              "per-matmul shared memory must fit in workgroup memory");
  }
  // If any operand has sub-byte elements, the constraint is skipped (v0).

  return success();
}

// TODO(#23535): These constraints intentionally mirror the Python tuner
// bug-for-bug.
// Do not change solver semantics here unless the Python constraints change
// first. Known follow-ups include materializing col-major MMA attrs from the
// qk_acc/pv_rhs layout match, moving gfx942 defaults (max_tile=512,
// num_subgroups=4, vector load width=128) into target data, and tightening
// sub-byte/div-zero schedule validity handling.

/// Emit attention constraints for a single root op under VectorDistribute.
/// v0: gates on `IREE::LinalgExt::OnlineAttentionOp`; static-shape only.
static LogicalResult
emitVectorDistributeAttentionConstraintsForOp(Operation *rootOp,
                                              RootOpAttr rootOpAttr) {
  auto attnOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(rootOp);
  if (!attnOp) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  FailureOr<AttentionDimInfo> dims = inferAttentionDimInfo(attnOp);
  if (failed(dims)) {
    // Don't propagate this as a compile error -- many shapes that the
    // tuner doesn't yet support (multi-dim M/N/K1/K2 roles, dynamic
    // inner dims, malformed indexing maps) still reach codegen and
    // should fall through to the default heuristics. Surface the miss
    // only in -debug-only=DEBUG_TYPE so unsupported shapes remain
    // non-fatal while still being debuggable with `iree-compile
    // --mlir-disable-threading -debug-only=...`.
    LDBG() << "attention constraint emitter: skipping " << *rootOp
           << ": inferAttentionDimInfo failed (likely multi-dim role, dynamic "
              "inner dim, or unsupported indexing maps)";
    return success();
  }

  // Compatible MMAs for QK: Q x K^T -> intermediate (f32 accumulator,
  // following the Python tuner -- `qk_matmul.acc_type` is typically f32).
  // The intermediate type lives nowhere on the attention op as a result;
  // the Python tuner uses f32 for the QK accumulator. We mirror that
  // choice here.
  MLIRContext *ctx = rootOp->getContext();
  Type qkAccType = Float32Type::get(ctx);
  SmallVector<Attribute> compatibleQKMMAs = getCompatibleAttentionMMAAttrs(
      gpuTarget, dims->mSize, dims->k2Size, dims->k1Size, dims->queryElemType,
      dims->keyElemType, qkAccType);
  if (compatibleQKMMAs.empty()) {
    return success();
  }

  // Compatible MMAs for PV: intermediate x V -> output. The Python dispatch
  // parser models the post-softmax probability as the V element type for the
  // PV matmul LHS and RHS; mirror that here for bug-for-bug parity.
  SmallVector<Attribute> compatiblePVMMAs = getCompatibleAttentionMMAAttrs(
      gpuTarget, dims->mSize, dims->nSize, dims->k2Size, dims->valueElemType,
      dims->valueElemType, dims->outputElemType);
  if (compatiblePVMMAs.empty()) {
    return success();
  }

  OpBuilder builder(ctx);
  DictionaryAttr knobs = buildVectorDistributeAttentionKnobsDict(
      ctx, *dims, compatibleQKMMAs, compatiblePVMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);
  // Reuse the generic shell builder; attention's indexing maps come
  // straight from the op interface.
  SmallVector<AffineMap> indexingMaps = {
      attnOp.getQueryMap(), attnOp.getKeyMap(), attnOp.getValueMap(),
      attnOp.getOutputMap()};
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               dims->domainRank, indexingMaps);

  return emitVectorDistributeAttentionConstraints(
      builder, attnOp, *dims, gpuTarget, shell.smtDimArgs, compatibleQKMMAs,
      compatiblePVMMAs);
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
