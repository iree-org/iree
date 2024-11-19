// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h"
#include <optional>

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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

// Metadata for a swizzle, that is, an (expand_shape -> transposition)
// pair of ops performing a change of layout within the tiles. This is used
// on GPU, where the tiles themselves can have an arbitrary layout.
struct TileSwizzle {
  struct Dim {
    // Describes what varies across this dimension.
    enum class Kind : int8_t {
      // This dimension is internal to one intrinsic on one thread. This
      // is only seen for intrinsic operands that are themselves vectors.
      // For example, with AMD MFMA, for the MFMA_F32_16x16x4_F32 intrinsic,
      // the C-matrix operand is a vector of 4 floats already at the level of
      // one intrinsic on one thread. That dimension of size 4 is 'Internal'.
      Internal,
      // This dimension is internal to one intrinsic, but is across threads.
      // For example, with AMD MFMA, for the MFMA_F32_16x16x4_F32 intrinsic,
      // the A-matrix tile has shape 16x4, and these two dimensions of size 16
      // and 4 are 'CrossThread': neither is visible at the single-thread level
      // (in the intrinsic itself, the A-matrix operand is a single scalar) but
      // as we move along these dimensions, we are moving over the 64 threads
      // of the subgroup.
      //
      // Another example of cross-thread dimensions is in kernels that are
      // "unrolled" across subgroups. Such dimensions are cross-subgroup, so in
      // particular they are cross-thread.
      CrossThread,
      // This dimensions is across intrinsics, as in, actual instructions in the
      // generated code. In other words, it is an actual unrolling factor,
      // resulting in this many more instructions being generated and executed
      // on each thread/subgroup.
      CrossIntrinsic
    };

    Kind kind = Kind::Internal;

    // The size of the dimension.
    int16_t size = 0;

    // Support constructing from any size type.
    template <typename T>
    Dim(Kind kind, T size) : kind(kind), size(size) {}
  };

  using ExpandShapeDimVectorType = llvm::SmallVector<Dim, 4>;
  using ExpandShapeType = llvm::SmallVector<ExpandShapeDimVectorType>;

  // This vector-of-vectors contains all the information needed to generate
  // a `tensor.expand_shape` creating additional internal dimensions into the
  // tile. For example, expandShape = [[16], [4, 2]] means that the original
  // tile shape [16, 8] gets expanded such that the first dimension 16 is left
  // unchanged, and the second dimension 8 gets split into two internal dims
  // of size 4 and 2.
  ExpandShapeType expandShape;
  // This permutation vector applies to the expanded dimensions and is used
  // to generate a `linalg.transpose` changing the layout of the tile. For
  // example, permutation[0] dictates which of the expanded dimensions becomes
  // the leading dimension of the layout.
  llvm::SmallVector<int64_t> permutation;
};

/// Container of information needed to materialize the layout transformations.
struct MaterializeEncodingInfo {
  // The next 3 fields are used to create a `tensor.pack` or `tensor.unpack` op,
  // changing the overall layout between row-major and tiled (where each tile is
  // row-major).
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;

  // The optional swizzle, see the comment on TileSwizzle. Only used on GPU.
  std::optional<TileSwizzle> swizzle;
};

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

static Attribute dimToArrayAttr(MLIRContext *ctx, TileSwizzle::Dim dim) {
  Builder b(ctx);
  return b.getDenseI16ArrayAttr({static_cast<int16_t>(dim.kind), dim.size});
}

static DictionaryAttr serializeTileSwizzle(MLIRContext *ctx,
                                           TileSwizzle swizzle) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;

  SmallVector<Attribute> expandShape;
  for (auto expandConfig : swizzle.expandShape) {
    Attribute expandAttr = b.getArrayAttr(
        llvm::map_to_vector(expandConfig, [&](TileSwizzle::Dim dim) {
          return dimToArrayAttr(ctx, dim);
        }));
    expandShape.push_back(expandAttr);
  }

  items.emplace_back(b.getStringAttr("expandShape"),
                     b.getArrayAttr(expandShape));
  items.emplace_back(b.getStringAttr("permutation"),
                     b.getI64ArrayAttr(swizzle.permutation));

  return b.getDictionaryAttr(items);
}

static std::optional<TileSwizzle> deserializeTileSwizzle(DictionaryAttr attr) {
  TileSwizzle swizzle;

  auto expandShapeAttr = attr.getNamed("expandShape");
  if (!expandShapeAttr) {
    return std::nullopt;
  }
  auto expandShapeArrayAttr = dyn_cast<ArrayAttr>(expandShapeAttr->getValue());
  if (!expandShapeArrayAttr) {
    return std::nullopt;
  }

  for (auto expandConfig : expandShapeArrayAttr.getAsRange<ArrayAttr>()) {
    // FIXME: The attribute that created by the builder can't be casted back to
    // the original attribute. Because it creates an ArrayAttr but not the
    // attribute that we specificed. It is not used in the prototype. For the
    // fix, see examples in the deserializeMaterializeEncodingInfo method.
    TileSwizzle::ExpandShapeDimVectorType vec;
    for (auto dimAttr : expandConfig.getAsRange<DenseI16ArrayAttr>()) {
      ArrayRef<int16_t> dimRef = dimAttr.asArrayRef();
      TileSwizzle::Dim dim(static_cast<TileSwizzle::Dim::Kind>(dimRef[0]),
                           dimRef[1]);
      vec.push_back(dim);
    }
    swizzle.expandShape.push_back(vec);
  }

  auto permutation =
      cast<DenseI64ArrayAttr>(attr.getNamed("permutation")->getValue())
          .asArrayRef();
  swizzle.permutation.assign(permutation.begin(), permutation.end());

  return swizzle;
}

static DictionaryAttr
serializeMaterializeEncodingInfo(MLIRContext *ctx,
                                 MaterializeEncodingInfo info) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;
  items.emplace_back(b.getStringAttr("innerDimsPos"),
                     b.getI64ArrayAttr(info.innerDimsPos));
  items.emplace_back(b.getStringAttr("innerTileSizes"),
                     b.getI64ArrayAttr(info.innerTileSizes));
  items.emplace_back(b.getStringAttr("outerDimsPerm"),
                     b.getI64ArrayAttr(info.outerDimsPerm));
  if (info.swizzle) {
    items.emplace_back(b.getStringAttr("swizzle"),
                       serializeTileSwizzle(ctx, info.swizzle.value()));
  }

  return b.getDictionaryAttr(items);
}

static std::optional<MaterializeEncodingInfo>
deserializeMaterializeEncodingInfo(DictionaryAttr attr) {
  MaterializeEncodingInfo info;

  auto innerDimsPosAttr = attr.getNamed("innerDimsPos");
  if (!innerDimsPosAttr) {
    return std::nullopt;
  }
  auto innerDimsPos = cast<ArrayAttr>(innerDimsPosAttr->getValue())
                          .getAsValueRange<IntegerAttr>();
  for (auto val : innerDimsPos) {
    info.innerDimsPos.push_back(val.getSExtValue());
  }

  auto innerTileSizesAttr = attr.getNamed("innerTileSizes");
  if (!innerTileSizesAttr) {
    return std::nullopt;
  }
  auto innerTileSizes = cast<ArrayAttr>(innerTileSizesAttr->getValue())
                            .getAsValueRange<IntegerAttr>();
  for (auto val : innerTileSizes) {
    info.innerTileSizes.push_back(val.getSExtValue());
  }

  auto outerDimsPermAttr = attr.getNamed("outerDimsPerm");
  if (!outerDimsPermAttr) {
    return std::nullopt;
  }
  auto outerDimsPerm = cast<ArrayAttr>(outerDimsPermAttr->getValue())
                           .getAsValueRange<IntegerAttr>();
  for (auto val : outerDimsPerm) {
    info.outerDimsPerm.push_back(val.getSExtValue());
  }

  if (attr.contains("swizzle")) {
    info.swizzle = deserializeTileSwizzle(
        cast<DictionaryAttr>(attr.getNamed("swizzle")->getValue()));
  }

  return info;
}

MaterializeEncodingInfo
getEncodingInfoForMatmul(IREE::Encoding::EncodingAttr encoding,
                         TileMxNxK tileMxNxK) {
  MaterializeEncodingInfo encodingInfo;
  auto cDims = getEncodingContractionDims(encoding);
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() == 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  std::optional<unsigned> batchDim =
      cDims->batch.empty() ? std::nullopt
                           : encoding.mapDimToOperandIndex(cDims->batch[0]);
  std::optional<unsigned> mDim =
      cDims->m.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->m[0]);
  std::optional<unsigned> nDim =
      cDims->n.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims->k[0]);
  if (batchDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(batchDim.value());
  }
  if (mDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(mDim.value());
    encodingInfo.innerDimsPos.push_back(mDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (nDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(nDim.value());
    encodingInfo.innerDimsPos.push_back(nDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (kDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(kDim.value());
    encodingInfo.innerDimsPos.push_back(kDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

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

  std::optional<MaterializeEncodingInfo> info =
      deserializeMaterializeEncodingInfo(getConfig());
  if (!info) {
    return mulAll(builder, loc, shape);
  }

  AffineExpr expr = builder.getAffineDimExpr(0);
  for (auto [pos, size] :
       llvm::zip_equal(info->innerDimsPos, info->innerTileSizes)) {
    shape[pos] = affine::makeComposedFoldedAffineApply(
        builder, loc, expr.ceilDiv(size) * size, {shape[pos]});
  }

  return mulAll(builder, loc, shape);
}

Encoding::EncodingSolverInterfaceAttr
CPUEncodingSolverAttr::cloneWithConfig(DictionaryAttr attr) const {
  return CPUEncodingSolverAttr::get(getContext(), attr);
}

Encoding::EncodingSolverInterfaceAttr
CPUEncodingSolverAttr::cloneWithSimplifiedConfig(Attribute attr) const {
  auto encoding = llvm::dyn_cast<IREE::Encoding::EncodingAttr>(attr);
  if (!encoding) {
    return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(*this);
  }

  SmallVector<TileMxNxK> enumeratedTileMxNxK =
      enumerateMatmulTile(encoding.getElementTypesArray(), getConfig());
  auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
  TileMxNxK chosenTileMxNxK = chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
  MaterializeEncodingInfo info =
      getEncodingInfoForMatmul(encoding, chosenTileMxNxK);

  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      CPUEncodingSolverAttr::get(
          getContext(), serializeMaterializeEncodingInfo(getContext(), info)));
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

  std::optional<MaterializeEncodingInfo> info =
      deserializeMaterializeEncodingInfo(getConfig());
  if (!info) {
    return mulAll(builder, loc, shape);
  }

  AffineExpr expr = builder.getAffineDimExpr(0);
  for (auto [pos, size] :
       llvm::zip_equal(info->innerDimsPos, info->innerTileSizes)) {
    int64_t padSize = 16;
    if (!ShapedType::isDynamic(size)) {
      padSize = size;
    }
    shape[pos] = affine::makeComposedFoldedAffineApply(
        builder, loc, expr.ceilDiv(padSize) * padSize, {shape[pos]});
  }

  return mulAll(builder, loc, shape);
}

Encoding::EncodingSolverInterfaceAttr
VMVXEncodingSolverAttr::cloneWithConfig(DictionaryAttr attr) const {
  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      VMVXEncodingSolverAttr::get(getContext(), attr));
}

Encoding::EncodingSolverInterfaceAttr
VMVXEncodingSolverAttr::cloneWithSimplifiedConfig(Attribute attr) const {
  auto encoding = llvm::dyn_cast<IREE::Encoding::EncodingAttr>(attr);
  if (!encoding) {
    return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(*this);
  }
  auto cDims = getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(*this);
  }

  Builder builder(getContext());
  SmallVector<TileMxNxK> enumeratedTileMxNxK;
  bool hasUkernelSupport = false;
  if (auto ukernelAttr = getTargetConfiguration().getNamed(
          builder.getStringAttr("ukernels"))) {
    auto strAttr = llvm::dyn_cast<StringAttr>(ukernelAttr->getValue());
    if (strAttr && strAttr.getValue() == "all") {
      hasUkernelSupport = true;
    }
  }
  if (!cDims->batch.empty()) {
    hasUkernelSupport = false;
  }

  if (hasUkernelSupport) {
    enumeratedTileMxNxK.push_back(TileMxNxK{
        ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic});
  } else {
    enumeratedTileMxNxK.push_back(
        TileMxNxK{8, 8, 4}); // Some vaguely reasonable tile shape.
    enumeratedTileMxNxK.push_back(
        TileMxNxK{4, 8, 4}); // Truncation of the above.
    enumeratedTileMxNxK.push_back(
        TileMxNxK{2, 8, 4}); // Truncation of the above.
    enumeratedTileMxNxK.push_back(
        TileMxNxK{1, 8, 4}); // Truncation of the above.
  }

  auto narrowDim = IREE::Encoding::getMatmulNarrowDim(encoding);
  TileMxNxK chosenTileMxNxK = chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
  LLVM_DEBUG(llvm::dbgs() << "chosenTileMxNxK: "; llvm::interleaveComma(
                 ArrayRef<int64_t>{chosenTileMxNxK.M, chosenTileMxNxK.N,
                                   chosenTileMxNxK.K},
                 llvm::dbgs());
             llvm::dbgs() << "\n");

  MaterializeEncodingInfo info =
      getEncodingInfoForMatmul(encoding, chosenTileMxNxK);

  return llvm::cast<Encoding::EncodingSolverInterfaceAttr>(
      VMVXEncodingSolverAttr::get(
          getContext(), serializeMaterializeEncodingInfo(getContext(), info)));
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
