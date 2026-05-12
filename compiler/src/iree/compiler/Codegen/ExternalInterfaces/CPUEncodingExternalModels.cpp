// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- CPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the following interfaces for CPU backends and the VMVX
// backend:
//
// - IREE::Encoding::LayoutResolverAttr
// - IREE::Encoding::SerializableAttr
// - IREE::Encoding::LayoutMaterializerAttr
// - IREE::Codegen::PackedLayoutMaterializerAttr
// - VerifiableTensorEncoding
//
// In these backends, we transpose narrow-N into narrow-M
// for a combination of reasons:
//
//   1. As linalg.matmul materializes into linalg.mmt4d, which has a transposed
//      RHS and therefore LHS<->RHS symmetry, transposeNarrowN is easy to
//      implement at that level.
//   2. We use ukernels, and this allows writing 2x fewer narrow ukernels.
//   3. Heuristics for cache-friendly dispatch tiling can get complex on CPU,
//      so it is nice that they have fewer narrow cases to consider.
//
// The only current exception to this is Arm SVE. It currently adheres to the
// canonical form of scalable vectorisation and keeps the N dimension to be
// scalable.
// This transposition is made easier by (and was all along part of the idea in)
// the RHS-transposition in mmt4d (the t in mmt4d), as generally with matrix
// multiplication
//
//   B * Transpose(A) == Transpose( A * Transpose(B) )
//
// so in mmt4d terms
//
//   mmt4d(B, A) == Transpose(mmt4d(A, B))
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler::IREE::CPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

namespace {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static FailureOr<IREE::Codegen::ScalableTileFlags>
getScalableTileFlags(linalg::ContractionDimensions cDims,
                     IREE::Encoding::EncodingAttr encoding,
                     DictionaryAttr config) {
  // TODO(egebeysel): I think this isScalable*Enabled flag should be temporary
  // and the temporary SME flag should probably come next to it.
  if (!isAArch64(config) || !isScalableVectorizationEnabled()) {
    LDBG() << "Pre-conditions to enable scalable tiling are not met!";
    return failure();
  }

  std::optional<unsigned> mDim =
      cDims.m.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.m[0]);
  std::optional<unsigned> nDim =
      cDims.n.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims.k[0]);
  IREE::Codegen::ScalableTileFlags scalableTiles;
  // TODO(egebeysel): Add logic for SME.
  if (mDim.has_value()) {
    if (hasFeature(config, "+sme")) {
      LDBG() << "SME with data-tiling is not supported yet!";
      return failure();
    }
    scalableTiles.push_back(false);
  }
  if (nDim.has_value()) {
    scalableTiles.push_back(hasFeature(config, "+sve") ||
                            hasFeature(config, "+sve2"));
  }
  if (kDim.has_value()) {
    scalableTiles.push_back(false);
  }
  return scalableTiles;
}

static void transposeInPlace(MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // Not a vector case, so all three arrays in `info` have size at least 2,
  // outerDimsPerm may have size 3 if there is a batch dimension, but in all
  // cases, the last 2 entries of each array are M and N, not batch.
  auto transpose = [](auto &a) { std::swap(a[a.size() - 2], a[a.size() - 1]); };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
  if (info.scalableTiles) {
    transpose(info.scalableTiles.value());
  }
}

static RankedTensorType
getExpandedType(RankedTensorType type, bool isBatched, bool isTransposed,
                SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

/// Given an input Value and a desired output element type, create and return
/// an element-wise linalg::GenericOp that extends the input Value to the
/// output element type. Returns `input` if casting is not needed.
static Value createElementWiseExtUIOp(OpBuilder &builder, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType() == outElemType) {
    return input;
  }
  SmallVector<AffineMap> maps(
      2, builder.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init =
      tensor::EmptyOp::create(builder, loc, inputMixedSizes, outElemType);
  return linalg::GenericOp::create(
             builder, loc, castedType, input, init, maps, iteratorTypes,
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Value castRes =
                   arith::ExtUIOp::create(b, nestedLoc, outElemType, args[0])
                       ->getResult(0);
               linalg::YieldOp::create(b, nestedLoc, castRes);
             })
      .getResult(0);
}

/// If needed, expand and the input Value, and return the resulting input with
/// the canonical mmt4d input shape. If the input element type is unsigned,
/// create a producer Linalg::GenericOp on the input that unsigned extends the
/// input to the output element type. This extension is required to keep the
/// unsignedness information on the input for ukernels. If `transpose` is true,
/// the `linalgOp`'s indexing maps are transposed.
static Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                             bool transpose, OpBuilder &builder,
                             SmallVectorImpl<ReassociationIndices> &ri,
                             ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && (transpose ^ cDims->n.empty()), ri);
    expandedValue =
        tensor::ExpandShapeOp::create(builder, loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(builder, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
}

/// Returns the best TileMxNxK from `enumeratedTiles` pool. If the
/// `hostDefinedUpperBound` is not empty, the chosen tile sizes can not be
/// greater than the values.
TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                           IREE::Encoding::MatmulNarrowDim narrowDim) {
  // Handle narrow-N by transposing to reduce to narrow-M. Note: the
  // enumeratedTiles currently only enumerate narrow-M cases.
  if (narrowDim.isN()) {
    narrowDim.dim = IREE::Encoding::MatmulNarrowDim::Dim::M;
    TileMxNxK tile = chooseMatmulTile(enumeratedTiles, narrowDim);
    std::swap(tile.M, tile.N);
    return tile;
  }
  // Handle kDynamic: currently this is only used with VMVX, where there is only
  // one enumerated tile and it has all three M/N/K dimensions dynamic, so for
  // now we only support that. Generalize that as needed when more dynamic tile
  // sizes are used outside of VMVX, e.g. perhaps some day with Arm SVE. Decide
  // how to incorporate the handling of kDynamic in the cost-model evaluation
  // below to decide when to prefer a dynamic vs a static tile shape.
  for (auto tile : enumeratedTiles) {
    if (ShapedType::isDynamic(tile.M) || ShapedType::isDynamic(tile.N) ||
        ShapedType::isDynamic(tile.K)) {
      assert(enumeratedTiles.size() == 1);
      assert(ShapedType::isDynamic(tile.M) && ShapedType::isDynamic(tile.N) &&
             ShapedType::isDynamic(tile.K));
      return tile;
    }
  }
  // We're going to "rate" the enumerated tiles.
  struct RatedTileMxNxK : TileMxNxK {
    RatedTileMxNxK() {}
    RatedTileMxNxK(TileMxNxK tile) : TileMxNxK(tile) {}
    // Penalize tiles that are wider in the M dimension than matmulNarrowM.
    int64_t paddingPenalty = 0;
    // Favor larger tiles, as long as they still minimize paddingPenalty.
    int64_t productMxNxK = 0;
  };
  SmallVector<RatedTileMxNxK> ratedTiles;
  ratedTiles.reserve(enumeratedTiles.size());
  int64_t bestPaddingPenalty = INT64_MAX;
  for (auto tile : enumeratedTiles) {
    RatedTileMxNxK ratedTile(tile);
    ratedTile.paddingPenalty = 0;
    // If we are choosing a tile for a narrow-M case, we want to minimize
    // padding along the M dimension.
    // The PowerOf2Ceil is so that we are OK with padding up to the next
    // power of two, we just try to avoid padding beyond that. For example,
    // if matmulNarrowM==7 and we have enumerated tiles with M=8,4,2,1, we
    // are OK with the tile that has M==8 even though it requires some padding.
    // Otherwise, we would be penalizing the tiles with M==8,4,2 and we would
    // end up selecting the vecmat tile (M==1) for that case!
    if (narrowDim) {
      ratedTile.paddingPenalty =
          std::max<int64_t>(tile.M - llvm::PowerOf2Ceil(narrowDim.size), 0);
    }
    ratedTile.productMxNxK = tile.M * tile.N * tile.K;
    ratedTiles.push_back(ratedTile);
    LDBG() << "candidate: "
           << llvm::interleaved(ArrayRef{tile.M, tile.N, tile.K})
           << " penalty:" << ratedTile.paddingPenalty;
    bestPaddingPenalty = std::min(bestPaddingPenalty, ratedTile.paddingPenalty);
  }
  RatedTileMxNxK bestRatedTile;
  for (auto ratedTile : ratedTiles) {
    // Choose only among tiles that minimize paddingPenalty. Among those,
    // maximize productMxNxK.
    if (ratedTile.paddingPenalty == bestPaddingPenalty &&
        bestRatedTile.productMxNxK < ratedTile.productMxNxK) {
      bestRatedTile = ratedTile;
    }
  }
  // Sanity check. This assert can only fail if there's a programming mistake
  // locally here.
  assert(bestRatedTile.paddingPenalty == bestPaddingPenalty);
  LDBG() << "bestRatedTile: "
         << llvm::interleaved(
                ArrayRef{bestRatedTile.M, bestRatedTile.N, bestRatedTile.K})
         << " penalty:" << bestRatedTile.paddingPenalty;
  return bestRatedTile;
}

static bool getEnableInnerTiledFromConfig(DictionaryAttr config) {
  Attribute attr = config.get("enable_inner_tiled");
  if (auto battr = dyn_cast_if_present<BoolAttr>(attr)) {
    return battr.getValue();
  }
  return false;
}

/// Returns the matmul {M, N, K} tile shape covered by a CPU DataTiledMMAAttr.
/// Derived directly from `getUndistributedTileTypes`: LHS is M×K (always) and
/// RHS is N×K (always), regardless of the `transposed_intrinsic` flag. The
/// ACC layout varies (M×N row-major vs. N×M column-major for transposed
/// intrinsics), so we derive M and N from the LHS and RHS tiles instead.
static IREE::Codegen::TileMxNxK getTileMxNxK(IREE::CPU::DataTiledMMAAttr mma) {
  SmallVector<VectorType> tiles;
  mma.getUndistributedTileTypes(tiles);
  assert(tiles.size() == 3 && "Expected LHS, RHS, ACC tile types");
  ArrayRef<int64_t> lhsShape = tiles[0].getShape();
  ArrayRef<int64_t> rhsShape = tiles[1].getShape();
  return IREE::Codegen::TileMxNxK{lhsShape[0], rhsShape[0], lhsShape[1]};
}

/// Returns the set of `+feature` strings that must all be present in the
/// target config for `intr` to be usable. Most AVX-512 intrinsics map
/// 1:1 to a single `+feature`, but the AVX2 f32 MMA intrinsics require both
/// `+avx2` and `+fma` (FMA3 is a separate ISA extension from AVX2 itself,
/// and our AVX2 intrinsics lower to `vfmadd...`).
static SmallVector<StringRef>
getMmaIntrinsicRequiredFeatures(IREE::CPU::MMAIntrinsic intr) {
  using IREE::CPU::MMAIntrinsic;
  switch (intr) {
  case MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32:
    return {"+avx2", "+fma"};
  case MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16:
    return {"+avx512f"};
  case MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16:
    return {"+avx512fp16"};
  case MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16:
    return {"+avx512bf16"};
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16:
  case MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16:
    return {"+avx512vnni"};
  default:
    return {};
  }
}

/// Returns the `MMA_GENERIC_SCALAR_1x1x1_REG*` variant for `config`. Almost
/// every 64-bit CPU has at least 16 architectural registers we could put
/// scalar tiles into (GPRs or lane 0 of a SIMD register, depending on what
/// LLVM picks), so we pick `_REG16` on 64-bit ISAs and `_REG8` on 32-bit
/// ones. This is a performance heuristic only — too high a budget would
/// spill, too low would underutilize — and the generic-scalar fallback is
/// slow either way.
static IREE::CPU::MMAIntrinsic
pickGenericScalarMMAForTarget(DictionaryAttr config) {
  using IREE::CPU::MMAIntrinsic;
  std::optional<llvm::Triple> triple = getTargetTriple(config);
  bool is64Bit = triple ? triple->isArch64Bit() : true;
  return is64Bit ? MMAIntrinsic::MMA_GENERIC_SCALAR_1x1x1_REG16
                 : MMAIntrinsic::MMA_GENERIC_SCALAR_1x1x1_REG8;
}

/// Returns the `MMAIntrinsic` cases potentially usable for `config`: the x86
/// architecture-specific intrinsics whose required ISA extensions are all
/// present, plus one of the architecture-agnostic type-polymorphic
/// `MMA_GENERIC_SCALAR_1x1x1_REG*` fallback variants (the one whose register
/// budget matches `config`'s target). Only the "natural" (M<=N) orientation
/// is listed; the M↔N-swapped orientation is expressed by the
/// `transposed_intrinsic` flag on DataTiledMMAAttr and is enumerated
/// separately by the cost model. The 1×1×1 generic intrinsic naturally
/// loses to any real intrinsic that fits, so it only wins as a fallback
/// when no real MMA covers the requested element types.
static SmallVector<IREE::CPU::MMAIntrinsic>
getMmaIntrinsicsForTargetConfig(DictionaryAttr config) {
  using IREE::CPU::MMAIntrinsic;
  if (!config) {
    return {};
  }
  // Always include the generic-scalar fallback first — it's the only
  // intrinsic guaranteed to apply on any target, so anchoring the list
  // with it means an early-return below can never accidentally produce
  // an empty result for an otherwise-valid target.
  SmallVector<MMAIntrinsic> out{pickGenericScalarMMAForTarget(config)};
  if (isX86(config)) {
    static const MMAIntrinsic kAllX86[] = {
        MMAIntrinsic::MMA_X86_AVX2_FMA_1x8x1_F32_F32,
        MMAIntrinsic::MMA_X86_AVX512_1x8x1_F64_F64,
        MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F32,
        MMAIntrinsic::MMA_X86_AVX512_1x16x1_F32_F16_CASTF32,
        MMAIntrinsic::MMA_X86_AVX512FP16_1x32x1_F16_F16,
        MMAIntrinsic::MMA_X86_AVX512BF16_1x16x2_F32_BF16,
        MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I16,
        MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I16,
        MMAIntrinsic::MMA_X86_AVX512_1x16x2_I32_I8_CASTI16,
        MMAIntrinsic::MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16,
    };
    for (MMAIntrinsic intr : kAllX86) {
      SmallVector<StringRef> required = getMmaIntrinsicRequiredFeatures(intr);
      if (required.empty()) {
        continue;
      }
      if (llvm::all_of(required,
                       [&](StringRef f) { return hasFeature(config, f); })) {
        out.push_back(intr);
      }
    }
  }
  return out;
}

/// Shape and element-bit-width summary of one intrinsic, with unroll factors
/// all 1, in a specific orientation. The LHS is (M, K), the RHS is (N, K) —
/// the `transposed` flag has already swapped M↔N inside the base attr.
/// Scalable dims (e.g. SVE) are treated as their 128-bit-minimum static
/// shape here; `getRegisterSpaceBytes` applies the matching simplification
/// on the capacity side.
struct IntrinsicInfo {
  int64_t intrinsicM = 0, intrinsicN = 0, intrinsicK = 0;
  int64_t lhsBits = 0, rhsBits = 0, accBits = 0;
};

/// Returns the IntrinsicInfo for `intr` in the given orientation, if its ABC
/// element types match `elementTypes`. Returns nullopt otherwise. The
/// `MMA_GENERIC_SCALAR_1x1x1_REG*` family is type-polymorphic — those
/// values match any element-type triple by construction; their `lhsBits` /
/// `rhsBits` / `accBits` stay at the struct's default 0 since the
/// generic-scalar branch of `chooseUnrolling` doesn't read them.
static std::optional<IntrinsicInfo>
getIntrinsicInfo(MLIRContext *ctx, ArrayRef<Type> elementTypes,
                 IREE::CPU::MMAIntrinsic intr, bool transposed) {
  if (IREE::CPU::isGenericScalar(intr)) {
    if (elementTypes.size() != 3 || !elementTypes[0] || !elementTypes[1] ||
        !elementTypes[2]) {
      return std::nullopt;
    }
    IntrinsicInfo info;
    info.intrinsicM = info.intrinsicN = info.intrinsicK = 1;
    return info;
  }
  auto base = IREE::CPU::DataTiledMMAAttr::get(
      ctx, intr, /*intrinsics_m=*/1, /*intrinsics_n=*/1,
      /*intrinsics_k=*/1, transposed, /*lhs_type=*/Type(),
      /*rhs_type=*/Type(), /*acc_type=*/Type());
  SmallVector<VectorType> baseTiles;
  base.getUndistributedTileTypes(baseTiles);
  if (baseTiles.size() != 3) {
    return std::nullopt;
  }
  if (baseTiles[0].getElementType() != elementTypes[0] ||
      baseTiles[1].getElementType() != elementTypes[1] ||
      baseTiles[2].getElementType() != elementTypes[2]) {
    return std::nullopt;
  }
  IntrinsicInfo info;
  info.intrinsicM = baseTiles[0].getShape()[0];
  info.intrinsicK = baseTiles[0].getShape()[1];
  info.intrinsicN = baseTiles[1].getShape()[0];
  info.lhsBits = baseTiles[0].getElementType().getIntOrFloatBitWidth();
  info.rhsBits = baseTiles[1].getElementType().getIntOrFloatBitWidth();
  info.accBits = baseTiles[2].getElementType().getIntOrFloatBitWidth();
  return info;
}

/// Phase 1 of `chooseCpuInnerTiledMmaForEncoding`: jointly pick the
/// intrinsic and its orientation (`transposed_intrinsic`). We maximize
/// `usefulOps` per invocation,
///   usefulOps = min(intrinsicM, M) * min(intrinsicN, N) * intrinsicK
/// where the `min` clamps an intrinsicM/intrinsicN that exceeds a narrow
/// static matmul dim — i.e. it charges the intrinsic for the padding it
/// would force. A dynamic matmul dim is treated as "not narrow" (the full
/// intrinsic size counts toward usefulOps). Ties are broken by the raw
/// intrinsic size `intrinsicM*intrinsicN*intrinsicK` (prefer the bigger
/// intrinsic: it gives the Phase-2 register budget more tiles to play with)
/// and, finally, by iteration order (non-transposed first). `K` is never
/// narrow for optimization purposes, so it never enters the `min`.
///
/// Returns nullopt if no compatible intrinsic exists.
static std::optional<std::pair<IREE::CPU::MMAIntrinsic, bool>>
chooseIntrinsic(MLIRContext *ctx, ArrayRef<Type> elementTypes,
                DictionaryAttr config,
                const IREE::Encoding::BxMxNxKxKb &matmulSizes) {
  auto usefulSize = [](int64_t matmulSize, int64_t intrinsicSize) -> int64_t {
    return ShapedType::isDynamic(matmulSize)
               ? intrinsicSize
               : std::min(intrinsicSize, matmulSize);
  };
  std::optional<std::pair<IREE::CPU::MMAIntrinsic, bool>> best;
  std::pair<int64_t, int64_t> bestScore = {-1, -1};
  for (IREE::CPU::MMAIntrinsic intr : getMmaIntrinsicsForTargetConfig(config)) {
    for (bool transposed : {false, true}) {
      std::optional<IntrinsicInfo> info =
          getIntrinsicInfo(ctx, elementTypes, intr, transposed);
      if (!info) {
        continue;
      }
      int64_t usefulOps = usefulSize(matmulSizes.M, info->intrinsicM) *
                          usefulSize(matmulSizes.N, info->intrinsicN) *
                          info->intrinsicK;
      int64_t rawOps = info->intrinsicM * info->intrinsicN * info->intrinsicK;
      std::pair<int64_t, int64_t> score = {usefulOps, rawOps};
      if (score > bestScore) {
        bestScore = score;
        best = {intr, transposed};
      }
    }
  }
  return best;
}

/// Power-of-two cap on one unroll dim. For a static matmul extent
/// `matmulSize` divided by `intrinsicSize`, we floor the cover count to a
/// power of two. For dynamic dims we fall back to `fallback`, which callers
/// set to something at least as large as the register budget — that budget
/// itself terminates the enumeration below, so no tighter static cap is
/// needed.
static int64_t po2UnrollCap(int64_t matmulSize, int64_t intrinsicSize,
                            int64_t fallback) {
  if (ShapedType::isDynamic(matmulSize)) {
    return fallback;
  }
  uint64_t cover =
      std::max<int64_t>(1, llvm::divideCeil(matmulSize, intrinsicSize));
  return llvm::bit_floor(cover);
}

// Phase 2 of `chooseCpuInnerTiledMmaForEncoding`: for an already-chosen
// (intrinsic, transposed) pair, pick the largest power-of-two unroll
// factors (intrinsicsM, intrinsicsN) such that the three tiles
// (ACC + LHS + RHS) still fit in the target's register budget,
// breaking ties with arithmetic intensity (effM*effN)/(effM+effN) so
// approximately-square tiles win.
//
// "Register budget" depends on what the lowering will use:
//   * Architecture-specific SIMD intrinsics use the vector register file,
//     measured in bits, with element widths from `IntrinsicInfo` so a
//     wider element type costs proportionally more of the budget.
//   * The type-polymorphic `MMA_GENERIC_SCALAR_1x1x1_REG*` lowers to
//     scalar arithmetic, where one element occupies one register (a GPR
//     or lane 0 of a SIMD register, depending on what LLVM picks)
//     regardless of bit width. So the budget is in registers, with all
//     element "widths" treated as 1.
//
// Returns nullopt if no feasible unrolling exists. The returned tuple is
// (intrinsicsM, intrinsicsN, intrinsicsK). When the (im, in) search loop
// finds no candidate that fits the budget — possible for the generic
// intrinsic with a sub-byte LHS/RHS forcing intrinsics_k = 8 against an
// 8-register budget — we fall back to (1, 1) so we still return *some*
// valid unrolling. Some register spill is preferable to failing data-
// tiling outright.
static std::optional<std::tuple<int64_t, int64_t, int64_t>>
chooseUnrolling(MLIRContext *ctx, ArrayRef<Type> elementTypes,
                IREE::CPU::MMAIntrinsic intr, bool transposed,
                const IREE::Encoding::BxMxNxKxKb &matmulSizes) {
  std::optional<IntrinsicInfo> info =
      getIntrinsicInfo(ctx, elementTypes, intr, transposed);
  if (!info) {
    return std::nullopt;
  }
  int64_t intrinsicsK = 1;
  int64_t budget;
  int64_t accUnit, lhsUnit, rhsUnit;
  if (IREE::CPU::isGenericScalar(intr)) {
    // Real hardware MMA intrinsics bake whatever K-grouping they need into
    // the intrinsic itself. The generic-scalar fallback doesn't, so for
    // sub-byte LHS/RHS types we have to group enough K-elements per
    // contiguous block that a packed group is byte-addressable: pick the
    // smallest power of two K such that K*lhsBits and K*rhsBits are both
    // multiples of 8. K∈{1,2,4,8} is enough to cover every type from i8/f8
    // (K=1) down to i1 (K=8).
    int64_t lhsBits = elementTypes[0].getIntOrFloatBitWidth();
    int64_t rhsBits = elementTypes[1].getIntOrFloatBitWidth();
    while (intrinsicsK <= 8 && (((intrinsicsK * lhsBits) % 8) != 0 ||
                                ((intrinsicsK * rhsBits) % 8) != 0)) {
      intrinsicsK *= 2;
    }
    if (intrinsicsK > 8) {
      return std::nullopt;
    }
    // The budget — chosen at intrinsic-pick time as a function of the
    // target's pointer width — is encoded in the enum value itself.
    budget = IREE::CPU::getGenericScalarRegisterBudget(intr);
    accUnit = lhsUnit = rhsUnit = 1;
  } else {
    budget = IREE::CPU::getRegisterSpaceBytes(intr) * 8;
    accUnit = info->accBits;
    lhsUnit = info->lhsBits;
    rhsUnit = info->rhsBits;
  }
  int64_t capMPo2 = po2UnrollCap(matmulSizes.M, info->intrinsicM, budget);
  int64_t capNPo2 = po2UnrollCap(matmulSizes.N, info->intrinsicN, budget);
  int64_t accTerm = info->intrinsicM * info->intrinsicN * accUnit;
  int64_t lhsTerm = info->intrinsicM * info->intrinsicK * intrinsicsK * lhsUnit;
  int64_t rhsTerm = info->intrinsicN * info->intrinsicK * intrinsicsK * rhsUnit;
  std::optional<std::pair<int64_t, int64_t>> bestMN;
  double bestIntensity = -1.0;
  // Enumerate power-of-two intrinsicsM; for each, pick the largest feasible
  // power-of-two intrinsicsN under the budget and the static N cap. The
  // budget bounds im on its own (im*lhsTerm alone must be < budget), which
  // terminates the loop without any numRegs-style cap.
  for (int64_t im = 1; im <= capMPo2; im *= 2) {
    int64_t remaining = budget - im * lhsTerm;
    if (remaining <= 0) {
      break;
    }
    int64_t inMaxBudget = remaining / (im * accTerm + rhsTerm);
    uint64_t inCap = std::min<int64_t>(capNPo2, inMaxBudget);
    if (inCap < 1) {
      continue;
    }
    int64_t in = llvm::bit_floor(inCap);
    double effM = static_cast<double>(im) * info->intrinsicM;
    double effN = static_cast<double>(in) * info->intrinsicN;
    double intensity = effM * effN / (effM + effN);
    if (intensity > bestIntensity) {
      bestIntensity = intensity;
      bestMN = {im, in};
    }
  }
  // Fall back to (1, 1) if no (im, in) fit — see function comment.
  if (!bestMN) {
    bestMN = {1, 1};
  }
  return std::make_tuple(bestMN->first, bestMN->second, intrinsicsK);
}

// Picks a CPU `DataTiledMMAAttr` for `iree_codegen.inner_tiled` given an
// encoding and target config.
static IREE::CPU::DataTiledMMAAttr
chooseCpuInnerTiledMmaForEncoding(MLIRContext *ctx,
                                  IREE::Encoding::EncodingAttr encoding,
                                  DictionaryAttr config) {
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (elementTypes.size() != 3) {
    return {};
  }
  FailureOr<IREE::Encoding::BxMxNxKxKb> matmulSizes =
      IREE::Encoding::getEncodingContractionLikeSizes(encoding);
  if (failed(matmulSizes)) {
    return {};
  }
  std::optional<std::pair<IREE::CPU::MMAIntrinsic, bool>> intrChoice =
      chooseIntrinsic(ctx, elementTypes, config, *matmulSizes);
  if (!intrChoice) {
    return {};
  }
  auto [intr, transposed] = *intrChoice;
  std::optional<std::tuple<int64_t, int64_t, int64_t>> unroll =
      chooseUnrolling(ctx, elementTypes, intr, transposed, *matmulSizes);
  if (!unroll) {
    return {};
  }
  auto [intrinsicsM, intrinsicsN, intrinsicsK] = *unroll;
  // The type-polymorphic generic intrinsic doesn't bake element types into
  // its enum value; we have to pin them down on the attr itself so the
  // lowering and `getABCElementTypes(ctx, attr)` can read them back.
  Type lhsType, rhsType, accType;
  if (IREE::CPU::isGenericScalar(intr)) {
    lhsType = elementTypes[0];
    rhsType = elementTypes[1];
    accType = elementTypes[2];
  }
  return IREE::CPU::DataTiledMMAAttr::get(ctx, intr, intrinsicsM, intrinsicsN,
                                          intrinsicsK, transposed, lhsType,
                                          rhsType, accType);
}

/// Lowers a contraction under a `CPUEncodingResolverAttr` with
/// `enable_inner_tiled = true` to an `iree_codegen.inner_tiled` op whose kind
/// is a CPU `data_tiled_mma_layout`. Returns nullptr if no CPU MMA intrinsic
/// is available for the encoding/target.
static Operation *lowerContractionToInnerTiled(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  auto cDims = linalg::inferContractionDims(linalgOp);
  if (!cDims->batch.empty()) {
    LDBG() << "inner_tiled lowering: batched contraction not implemented";
    return nullptr;
  }

  MLIRContext *ctx = builder.getContext();
  Location loc = linalgOp.getLoc();
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  AffineExpr d2 = builder.getAffineDimExpr(2);
  // RHS uses (d1, d2) i.e. (N_iter, K_iter) outer ordering — mmt4d-style — to
  // match the packed RHS operand shape produced upstream (the RHS pack uses
  // `outer_dims_perm = [1, 0]` so its outer dims come out as N_iter, K_iter,
  // not K_iter, N_iter).
  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(3, 0, {d0, d2}, ctx),
      AffineMap::get(3, 0, {d1, d2}, ctx),
      AffineMap::get(3, 0, {d0, d1}, ctx),
  };
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};

  DictionaryAttr targetConfig;
  if (auto cpuResolver = dyn_cast<IREE::CPU::CPUEncodingResolverAttr>(
          cast<Attribute>(layoutAttr))) {
    targetConfig = cpuResolver.getConfiguration();
  }

  // Mirrors the GPU DataTiledMMA path: the MMA attribute is determined from
  // the encoding + target config alone (the same inputs used by the encoding
  // materialization side), not reverse-engineered from the already-packed
  // operand tile shapes.
  IREE::CPU::DataTiledMMAAttr chosenKind =
      chooseCpuInnerTiledMmaForEncoding(ctx, resultEncoding, targetConfig);
  if (!chosenKind) {
    LDBG() << "inner_tiled lowering: no CPU DataTiledMMA kind available for "
              "encoding/target";
    return nullptr;
  }

  auto semanticsAttr = IREE::CPU::InnerTiledSemanticsAttr::get(ctx);
  Operation *inner = IREE::Codegen::InnerTiledOp::create(
      builder, loc, ValueRange{operands[0], operands[1]},
      ValueRange{operands[2]}, indexingMaps, iteratorTypes, chosenKind,
      semanticsAttr);
  return inner;
}

Operation *lowerContractionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    encodingInfo = packedLayoutAttr.getEncodingInfo(
        cast<RankedTensorType>(linalgOp->getResultTypes()[0]));
  }

  if (isIdentityLayout(encodingInfo)) {
    return dropEncodingAndCloneOp(builder, linalgOp,
                                  operands.take_front(inputs.size()),
                                  operands.drop_front(inputs.size()));
  }

  bool transpose = isNarrowNResult(resultEncoding);
  // Do not transpose in case we have scalable tiles.
  transpose &= llvm::none_of(
      encodingInfo.scalableTiles.value_or(IREE::Codegen::ScalableTileFlags{}),
      [](bool flag) { return flag; });
  SmallVector<Type> elemTypes = lhsEncoding.getElementTypesArray();
  SmallVector<ReassociationIndices> ri;
  Value newLhs = getMmt4dOperand(operands[0], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/0);
  Value newRhs = getMmt4dOperand(operands[1], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/1);
  Value newResult = getMmt4dOperand(operands[2], linalgOp, transpose, builder,
                                    ri, elemTypes, /*operandIdx=*/2);
  if (transpose) {
    std::swap(newLhs, newRhs);
  }
  Type newResultType = newResult.getType();
  auto cDims = IREE::Encoding::getEncodingContractionDims(lhsEncoding);
  Operation *result;
  if (cDims->batch.empty()) {
    result = linalg::Mmt4DOp::create(builder, linalgOp.getLoc(), newResultType,
                                     ValueRange{newLhs, newRhs},
                                     ValueRange{newResult});
  } else {
    result = linalg::BatchMmt4DOp::create(
        builder, linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
        ValueRange{newResult});
  }
  if (!ri.empty()) {
    result = tensor::CollapseShapeOp::create(builder, linalgOp->getLoc(),
                                             operands[2].getType(),
                                             result->getResult(0), ri);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.cpu_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32(DictionaryAttr config) {
  if (hasUkernel(config)) {
    return {
        TileMxNxK{8, 8, 4}, // Some reasonable tile shape.
        TileMxNxK{4, 8, 4}, // Truncation of the above.
        TileMxNxK{2, 8, 4}, // Truncation of the above.
        TileMxNxK{1, 8, 4}, // Truncation of the above.
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}
// RISC-V has vector register length extensions: zvl128b, zvl256b etc.
// If these extension are specified in target cpu feature,
// they can be used to determine VLEN. This function assumes that
// 'v' feature is present
size_t getRISCVVVlenFromCPUFeatures(DictionaryAttr config) {
  // If +zvl* feature is not explicitly specified,
  // fallback to +zvl128b, as spec specifies minimum VLEN
  // of 128b for the V extension: https://rb.gy/p8rbzv
  size_t vlen;
  if (hasFeature(config, "+zvl65536b")) {
    vlen = 65536;
  } else if (hasFeature(config, "+zvl32768b")) {
    vlen = 32768;
  } else if (hasFeature(config, "+zvl16384b")) {
    vlen = 16384;
  } else if (hasFeature(config, "+zvl8192b")) {
    vlen = 8192;
  } else if (hasFeature(config, "+zvl4096b")) {
    vlen = 4096;
  } else if (hasFeature(config, "+zvl2048b")) {
    vlen = 2048;
  } else if (hasFeature(config, "+zvl1024b")) {
    vlen = 1024;
  } else if (hasFeature(config, "+zvl512b")) {
    vlen = 512;
  } else if (hasFeature(config, "+zvl256b")) {
    vlen = 256;
  } else {
    vlen = 128;
  }
  return vlen;
}
// Enumerate tile sizes to choose from on riscv64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv64(TypeRange elementTypes, DictionaryAttr config) {

  // Data-Tiling is only implemented for the V extension
  if (!hasFeature(config, "+v")) {
    return {};
  }
  size_t vlen = getRISCVVVlenFromCPUFeatures(config);
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    // VLEN-aware Tile size selection
    // One concern that needs to be addressed here is that
    // for larger VLENs tile sizes would be very large
    // leading to a very high padding overhead
    int N0 = vlen / 8;
    return {
        TileMxNxK{7, N0, 1}, // Aim to use vfmacc, 100% register utilization.
        TileMxNxK{4, N0, 1}, // Truncation of the above.
        TileMxNxK{2, N0, 1}, // Truncation of the above.
        TileMxNxK{1, N0, 1}, // Truncation of the above.
    };
  }
  if (lhs.isF16() && rhs.isF16()) {
    int N0 = vlen / 8;
    if (hasFeature(config, "+zvfh")) {
      return {
          TileMxNxK{7, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+zvfhmin")) {
      return {
          TileMxNxK{6, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileArm64(TypeRange elementTypes,
                                                       DictionaryAttr config) {
  // For SVE and scalable vectors, this methods selects base sizes that match
  // the NEON fixed-width sizes.
  // TODO: Add SME inner tile sizes and corresponding tests.
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(config, "+bf16")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use BFMMLA.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16, as we could typically have a 2x wider
      // tile in that case. However, on current CPUs, the existing tiles seem
      // wide enough already to approach peak performance.
      return {
          TileMxNxK{8, 8, 1}, // Aim to use FMLA or FMLAL.
          TileMxNxK{4, 8, 1}, // Truncation of the above.
          TileMxNxK{2, 8, 1}, // Truncation of the above.
          TileMxNxK{1, 8, 1}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use SDOT.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(4) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{4, 8, 16},
          TileMxNxK{2, 8, 16},
          TileMxNxK{1, 8, 16},
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 8},
          TileMxNxK{4, 8, 8},
          TileMxNxK{2, 8, 8},
          TileMxNxK{1, 8, 8},
      };
    }
    return {
        TileMxNxK{4, 16, 2},
        TileMxNxK{2, 16, 2},
        TileMxNxK{1, 16, 2},
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on x86-64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileX86_64(TypeRange elementTypes,
                                                        DictionaryAttr config) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(config, "+avx512bf16")) {
        return {
            TileMxNxK{16, 16, 2}, // Aim to use VDPBF16PS (zmm).
            TileMxNxK{8, 16, 2},  // Truncation of the above.
            TileMxNxK{4, 16, 2},  // Truncation of the above.
            TileMxNxK{2, 16, 2},  // Truncation of the above.
            TileMxNxK{1, 16, 2},  // Truncation of the above.
        };
      }
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16.
      if (hasFeature(config, "+avx512f")) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      }
      if (hasFeature(config, "+avx")) {
        // Note: for good performance, most +avx users will also want to add
        // +fma, but that's a local instruction selection detail and the tile
        // layout is unaffected, as there are enough registers even with the
        // need for intermediate product registers when +fma is not used.
        return {
            TileMxNxK{8, 8, 1}, // Aim to use VFMADD* (ymm).
            TileMxNxK{4, 8, 1}, // Truncation of the above.
            TileMxNxK{2, 8, 1}, // Truncation of the above.
            TileMxNxK{1, 8, 1}, // Truncation of the above.
        };
      }
      // SSE fallback.
      return {
          TileMxNxK{8, 4, 1}, // Aim to use MULPS/ADDPS (xmm).
          TileMxNxK{4, 4, 1}, // Truncation of the above.
          TileMxNxK{2, 4, 1}, // Truncation of the above.
          TileMxNxK{1, 4, 1}, // Truncation of the above.
      };
    }
  }

  if (out.isSignlessInteger(32) &&
      ((lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8)) ||
       (lhs.isSignlessInteger(16) && rhs.isSignlessInteger(16)))) {
    if (hasFeature(config, "+avx512vnni")) {
      // This is the same tile size as with VPMADDWD as the only difference
      // is that VPDPWSSD accumulates. VPDPBUSD would call for {16, 16, 4} but
      // we can't easily use it because of its unsigned*signed semantics.
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPDPWSSD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx2")) {
      return {
          TileMxNxK{8, 8, 2}, // Aim to use VPMADDWD (ymm).
          TileMxNxK{4, 8, 2}, // Truncation of the above.
          TileMxNxK{2, 8, 2}, // Truncation of the above.
          TileMxNxK{1, 8, 2}, // Truncation of the above.
      };
    }
    // SSE fallback.
    return {
        TileMxNxK{8, 4, 2}, // Aim to use PMADDWD (xmm).
        TileMxNxK{4, 4, 2}, // Truncation of the above.
        TileMxNxK{2, 4, 2}, // Truncation of the above.
        TileMxNxK{1, 4, 2}, // Truncation of the above.
    };
  }

  if (out.isSignlessInteger(32) && lhs.isSignlessInteger(16) &&
      rhs.isUnsignedInteger(4)) {
    // Experimental s16u4s32 case. Focusing only on the vecmat case for now.
    if (hasFeature(config, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static SmallVector<TileMxNxK>
enumerateCPUMatmulTiles(IREE::Encoding::EncodingAttr encoding,
                        DictionaryAttr config) {
  // Enumerate available tile shapes for the given encoding and config.
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (isAArch64(config)) {
    return enumerateMatmulTileArm64(elementTypes, config);
  }
  if (isX86_64(config)) {
    return enumerateMatmulTileX86_64(elementTypes, config);
  }
  if (isRISCV32(config)) {
    return enumerateMatmulTileRiscv32(config);
  }
  if (isRISCV64(config)) {
    return enumerateMatmulTileRiscv64(elementTypes, config);
  }
  return {};
}

struct CPUEncodingPackedLayoutMaterializerAttr
    : PackedLayoutMaterializerAttrExternalModelBase<
          CPUEncodingPackedLayoutMaterializerAttr, CPUEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<CPUEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    DictionaryAttr config = layoutAttr.getConfiguration();
    if (getEnableInnerTiledFromConfig(config)) {
      return getInnerTiledEncodingInfo(type.getContext(), encoding, *cDims,
                                       config);
    }
    return getMmt4dEncodingInfo(encoding, *cDims, config);
  }

private:
  /// Legacy mmt4d path: enumerate candidate `TileMxNxK`s, pick one with the
  /// narrow-dim-aware scoring, and apply the narrow-N→narrow-M transpose
  /// trick required by the handwritten microkernels.
  static MaterializeEncodingInfo
  getMmt4dEncodingInfo(IREE::Encoding::EncodingAttr encoding,
                       const linalg::ContractionDimensions &cDims,
                       DictionaryAttr config) {
    MaterializeEncodingInfo info;
    SmallVector<TileMxNxK> enumeratedTileMxNxK =
        enumerateCPUMatmulTiles(encoding, config);
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    FailureOr<IREE::Codegen::ScalableTileFlags> scalableFlags =
        getScalableTileFlags(cDims, encoding, config);
    if (succeeded(scalableFlags)) {
      info.scalableTiles = std::move(scalableFlags);
    }
    if (IREE::Encoding::isNarrowNResult(encoding) &&
        llvm::none_of(info.scalableTiles.value_or(Codegen::ScalableTileFlags{}),
                      [](bool flag) { return flag; })) {
      transposeInPlace(info);
    }
    return info;
  }

  /// Intrinsic-first path for `iree_codegen.inner_tiled`: pick a
  /// `DataTiledMMAAttr` (same decision the lowering will make) and derive the
  /// packed-layout tile shape from it. No narrow-N transpose: unlike the
  /// legacy mmt4d path, the inner_tiled lowering handles narrow dims natively
  /// via `intrinsics_m` / `intrinsics_n`.
  static MaterializeEncodingInfo getInnerTiledEncodingInfo(
      MLIRContext *ctx, IREE::Encoding::EncodingAttr encoding,
      const linalg::ContractionDimensions &cDims, DictionaryAttr config) {
    MaterializeEncodingInfo info;
    IREE::CPU::DataTiledMMAAttr mma =
        chooseCpuInnerTiledMmaForEncoding(ctx, encoding, config);
    if (!mma) {
      return info;
    }
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, getTileMxNxK(mma));
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    FailureOr<IREE::Codegen::ScalableTileFlags> scalableFlags =
        getScalableTileFlags(cDims, encoding, config);
    if (succeeded(scalableFlags)) {
      info.scalableTiles = std::move(scalableFlags);
    }
    return info;
  }
};

struct CPUEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          CPUEncodingResolverMaterializerAttr, CPUEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    // Scaled contraction (MX matmul) is not yet supported on CPU, so we drop
    // the encoding and clone the op as-is.
    if (IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
      int64_t numInputs = linalgOp.getNumDpsInputs();
      return dropEncodingAndCloneOp(b, linalgOp,
                                    convertedOperands.take_front(numInputs),
                                    convertedOperands.drop_front(numInputs));
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      DictionaryAttr config = layoutAttr.getConfiguration();
      if (getEnableInnerTiledFromConfig(config)) {
        return lowerContractionToInnerTiled(
            b, linalgOp, convertedOperands,
            cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
      }
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct CPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          CPULayoutResolverAttr, CPUEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    if (std::optional<StringRef> cpuFeatures = getConfigCpuFeatures(config)) {
      addConfigCpuFeatures(ctx, cpuFeatures.value(), configItems);
    }
    if (std::optional<StringRef> targetTriple = getConfigTargetTriple(config)) {
      addConfigTargetTriple(ctx, targetTriple.value(), configItems);
    }
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    storeNamedAttrIfPresent(configItems, config, "enable_inner_tiled");
    return CPUEncodingResolverAttr::get(ctx,
                                        DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return CPUEncodingResolverAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct CPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<CPUSerializableAttr,
                                                      CPUEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration = cast<CPUEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct CPUEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<CPUEncodingResolverVerifier,
                                                    CPUEncodingResolverAttr> {

  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.vmvx_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateVMVXMatmulTiles(linalg::ContractionDimensions cDims,
                         IREE::Encoding::EncodingAttr encoding,
                         DictionaryAttr config) {
  bool hasUkernelSupport = hasUkernel(config);

  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul. Also, they are not set up for narrow M/N matmul, so it is
  // disabled when it is the case.
  if (!cDims.batch.empty() || getPo2MatmulNarrowDim(encoding)) {
    hasUkernelSupport = false;
  }
  if (hasUkernelSupport) {
    // VMVX+ukernel uses dynamic tile shapes.
    return {TileMxNxK{ShapedType::kDynamic, ShapedType::kDynamic,
                      ShapedType::kDynamic}};
  }

  return {
      TileMxNxK{8, 8, 4}, // Some vaguely reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
}

struct VMVXEncodingPackedLayoutMaterializerAttr final
    : PackedLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingPackedLayoutMaterializerAttr, VMVXEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK = enumerateVMVXMatmulTiles(
        cDims.value(), encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumerated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    if (IREE::Encoding::isNarrowNResult(encoding)) {
      transposeInPlace(info);
    }
    return info;
  }
};

struct VMVXEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingResolverMaterializerAttr, VMVXEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct VMVXLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          VMVXLayoutResolverAttr, VMVXEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return VMVXEncodingResolverAttr::get(ctx,
                                         DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return VMVXEncodingResolverAttr::get(
        ctx, getPackedLayoutImpl(attr, type, /*addEncodingAttr=*/true));
  }
};

struct VMVXSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<
          VMVXSerializableAttr, VMVXEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct VMVXEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<
          VMVXEncodingResolverVerifier, VMVXEncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

} // namespace

void registerCPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::CPUEncodingResolverAttr::attachInterface<
            CPUEncodingPackedLayoutMaterializerAttr,
            CPUEncodingResolverMaterializerAttr, CPULayoutResolverAttr,
            CPUSerializableAttr, CPUEncodingResolverVerifier>(*ctx);
        IREE::CPU::VMVXEncodingResolverAttr::attachInterface<
            VMVXEncodingPackedLayoutMaterializerAttr,
            VMVXEncodingResolverMaterializerAttr, VMVXLayoutResolverAttr,
            VMVXSerializableAttr, VMVXEncodingResolverVerifier>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
