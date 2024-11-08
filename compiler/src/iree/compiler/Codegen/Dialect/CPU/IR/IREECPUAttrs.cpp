// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-cpu-attrs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::CPU {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static OpFoldResult mulAll(OpBuilder &builder, Location &loc,
                           ArrayRef<OpFoldResult> shape) {
  OpFoldResult res = builder.getIndexAttr(1);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  for (auto dimSize : shape) {
    res = affine::makeComposedFoldedAffineApply(builder, loc, d0 * d1,
                                                {res, dimSize});
  }
  return res;
}

//===----------------------------------------------------------------------===//
// iree_cpu.cpu_encoding_solver
//===----------------------------------------------------------------------===//

static bool hasFeature(DictionaryAttr config, StringRef feature) {
  if (!config.contains("cpu_features")) {
    return false;
  }
  auto cpuFeaturesAttr = config.getAs<StringAttr>("cpu_features");
  if (!cpuFeaturesAttr) {
    return false;
  }

  // Find feature string in list of features, making sure that we don't match a
  // sub-string.
  std::stringstream sstream(cpuFeaturesAttr.getValue().str());
  std::string str;
  while (std::getline(sstream, str, ',')) {
    if (str == feature) {
      return true;
    }
  }

  return false;
}

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

// TODO: Do we really need to know the target triple? Can we just look at the
// configuration like this?
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTile(TypeRange elementTypes,
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

static TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
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

    LLVM_DEBUG(llvm::dbgs() << "candidate: "; llvm::interleaveComma(
                   ArrayRef<int64_t>{tile.M, tile.N, tile.K}, llvm::dbgs());
               llvm::dbgs() << " penalty:" << ratedTile.paddingPenalty << "\n");

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
  return bestRatedTile;
}

OpFoldResult CPUEncodingSolverAttr::calculateStorageElementCountInBytes(
    OpBuilder &builder, RankedTensorType type, ValueRange dynamicDims) const {
  auto encoding =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());

  Location loc = builder.getUnknownLoc();
  SmallVector<OpFoldResult> shape =
      getMixedValues(type.getShape(), dynamicDims, builder);
  if (!encoding) {
    return mulAll(builder, loc, shape);
  }
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return mulAll(builder, loc, shape);
  }

  SmallVector<TileMxNxK> enumeratedTileMxNxK =
      enumerateMatmulTile(encoding.getElementTypesArray(), getConfig());
  if (enumeratedTileMxNxK.empty()) {
    return mulAll(builder, loc, shape);
  }
  auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
  TileMxNxK chosenTileMxNxK = chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
  LLVM_DEBUG(llvm::dbgs() << "[iree-cpu-attrs]: choose TileMxNxK: ["
                          << chosenTileMxNxK.M << ", " << chosenTileMxNxK.N
                          << ", " << chosenTileMxNxK.K << "], operand_index: "
                          << encoding.getOperandIndex() << "\n";);

  AffineExpr expr = builder.getAffineDimExpr(0);
  auto pad = [&](int64_t dim, int64_t val) -> void {
    std::optional<unsigned> maybeMappedDim = encoding.mapDimToOperandIndex(dim);
    if (!maybeMappedDim) {
      return;
    }
    unsigned mappedDim = maybeMappedDim.value();
    LLVM_DEBUG(llvm::dbgs()
                   << "Pad dim #" << dim << " with size=" << val << "\n";);
    shape[mappedDim] = affine::makeComposedFoldedAffineApply(
        builder, loc, expr.ceilDiv(val) * val, {shape[mappedDim]});
  };
  for (auto m : cDims->m) {
    pad(m, chosenTileMxNxK.M);
  }
  for (auto n : cDims->n) {
    pad(n, chosenTileMxNxK.N);
  }
  for (auto k : cDims->k) {
    pad(k, chosenTileMxNxK.K);
  }

  return mulAll(builder, loc, shape);
}

Encoding::EncodingSolverInterfaceAttr
CPUEncodingSolverAttr::cloneWithConfig(DictionaryAttr attr) const {
  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      CPUEncodingSolverAttr::get(getContext(), attr));
}

DictionaryAttr CPUEncodingSolverAttr::getConfig() const {
  return getTargetConfiguration();
}

//===----------------------------------------------------------------------===//
// iree_cpu.vmvx_encoding_solver
//===----------------------------------------------------------------------===//

OpFoldResult VMVXEncodingSolverAttr::calculateStorageElementCountInBytes(
    OpBuilder &builder, RankedTensorType type, ValueRange dynamicDims) const {
  auto encoding =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());

  Location loc = builder.getUnknownLoc();
  SmallVector<OpFoldResult> shape =
      getMixedValues(type.getShape(), dynamicDims, builder);
  if (!encoding) {
    return mulAll(builder, loc, shape);
  }
  // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return mulAll(builder, loc, shape);
  }

  bool hasUkernelSupport = false;
  if (getTargetConfiguration().get(builder.getStringAttr("ukernels"))) {
    hasUkernelSupport = true;
  }

  if (hasUkernelSupport) {
    AffineExpr expr = builder.getAffineDimExpr(0);
    auto padTo16 = [&](int64_t dim) -> void {
      std::optional<unsigned> maybeMappedDim =
          encoding.mapDimToOperandIndex(dim);
      if (!maybeMappedDim) {
        return;
      }
      unsigned mappedDim = maybeMappedDim.value();
      shape[mappedDim] = affine::makeComposedFoldedAffineApply(
          builder, loc, expr.ceilDiv(16) * 16, {shape[mappedDim]});
    };
    for (auto m : cDims->m) {
      padTo16(m);
    }
    for (auto n : cDims->n) {
      padTo16(n);
    }
    for (auto k : cDims->k) {
      padTo16(k);
    }
  }

  return mulAll(builder, loc, shape);
}

Encoding::EncodingSolverInterfaceAttr
VMVXEncodingSolverAttr::cloneWithConfig(DictionaryAttr attr) const {
  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      VMVXEncodingSolverAttr::get(getContext(), attr));
}

DictionaryAttr VMVXEncodingSolverAttr::getConfig() const {
  return getTargetConfiguration();
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREECPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::CPU
