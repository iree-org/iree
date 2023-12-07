// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/CPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTilesVMVX(EncodingUser user, ExecutableTargetAttr target) {
  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul.
  if (hasUkernel(target) && user != EncodingUser::BATCH_MATMUL) {
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

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32(EncodingUser user, ExecutableTargetAttr target) {
  if (hasUkernel(target)) {
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

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileArm64(EncodingUser user, TypeRange elementTypes,
                         ExecutableTargetAttr target) {
  if (user != EncodingUser::MATMUL && user != EncodingUser::BATCH_MATMUL) {
    return {};
  }

  // Data-tiling for SVE is not implemented yet.
  if (hasFeature(target, "+sve") || hasFeature(target, "+sve2")) {
    return {};
  }

  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(target, "+bf16")) {
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
    if (hasFeature(target, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(target, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use SDOT.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
    return {
        TileMxNxK{8, 8, 1}, // Aim to use SMLAL.
        TileMxNxK{4, 8, 1}, // Truncation of the above.
        TileMxNxK{2, 8, 1}, // Truncation of the above.
        TileMxNxK{1, 8, 1}, // Truncation of the above.
    };
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on x86-64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileX86_64(EncodingUser user, TypeRange elementTypes,
                          ExecutableTargetAttr target) {
  if (user != EncodingUser::MATMUL && user != EncodingUser::BATCH_MATMUL) {
    return {};
  }

  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(target, "+avx512bf16")) {
        return {
            TileMxNxK{16, 16, 2}, // Aim to use VDPBF16PS (zmm).
            TileMxNxK{8, 16, 2},  // Truncation of the above.
            TileMxNxK{4, 16, 2},  // Truncation of the above.
            TileMxNxK{2, 16, 2},  // Truncation of the above.
            TileMxNxK{1, 16, 2},  // Truncation of the above.
        };
      }
    }

    // The LHS and RHS can be floating or integer types here. For the
    // combinations where ukernel can't support, the codegen should still be
    // able to generate reasonable code for the data-tiling.

    // Note: 16-bit floating point types currently use the same tile size as
    // f32. This makes sense when either (1) the accumulator is f32, or (2)
    // the arithmetic will have to expand f16 to f32 in registers. We may
    // reconsider when taking advantage of native f16/bf16 arithmetic when the
    // accumulator itself is f16/bf16.
    if (hasFeature(target, "+avx512f")) {
      if (hasUkernel(target)) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      } else {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 32, 1},  // Use same number of accumulators.
            TileMxNxK{4, 64, 1},  // Use same number of accumulators.
            TileMxNxK{2, 64, 1},  // Use half the number of accumulators.
            TileMxNxK{1, 128, 1}, // Use half the number of accumulators.
        };
      }
    }
    if (hasFeature(target, "+avx")) {
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

  if (out.isSignlessInteger(32) &&
      ((lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8)) ||
       (lhs.isSignlessInteger(16) && rhs.isSignlessInteger(16)))) {
    if (hasFeature(target, "+avx512vnni")) {
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
    if (hasFeature(target, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(target, "+avx2")) {
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
    if (hasFeature(target, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                                  int64_t matmulNarrowM,
                                  int64_t matmulNarrowN) {
  // Handle narrow-N by transposing to reduce to narrow-M. Note: the
  // enumeratedTiles currently only enumerate narrow-M cases.
  if (matmulNarrowN && (!matmulNarrowM || matmulNarrowN < matmulNarrowM)) {
    TileMxNxK tile = chooseMatmulTile(enumeratedTiles, matmulNarrowN, 0);
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
    if (matmulNarrowM) {
      ratedTile.paddingPenalty =
          std::max<int64_t>(tile.M - llvm::PowerOf2Ceil(matmulNarrowM), 0);
    }
    ratedTile.productMxNxK = tile.M * tile.N * tile.K;
    ratedTiles.push_back(ratedTile);
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

SmallVector<TileMxNxK> enumerateMatmulTileMxNxK(EncodingUser user,
                                                TypeRange elementTypes,
                                                ExecutableTargetAttr target) {
  if (isVMVXBackend(target)) {
    return enumerateMatmulTilesVMVX(user, target);
  }
  if (isAArch64(target)) {
    return enumerateMatmulTileArm64(user, elementTypes, target);
  }
  if (isX86_64(target)) {
    return enumerateMatmulTileX86_64(user, elementTypes, target);
  }
  if (isRISCV32(target)) {
    return enumerateMatmulTileRiscv32(user, target);
  }
  return {};
}

struct CPUMaterializeEncodingPass
    : public CPUMaterializeEncodingBase<CPUMaterializeEncodingPass> {
  CPUMaterializeEncodingPass() : targetAttr(nullptr) {}
  explicit CPUMaterializeEncodingPass(IREE::HAL::ExecutableTargetAttr attr)
      : targetAttr(attr) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;

private:
  IREE::HAL::ExecutableTargetAttr targetAttr;
};

struct CPUMaterializeUpperBoundTileSizePass
    : public CPUMaterializeUpperBoundTileSizeBase<
          CPUMaterializeUpperBoundTileSizePass> {
  CPUMaterializeUpperBoundTileSizePass() = default;
  explicit CPUMaterializeUpperBoundTileSizePass(
      ArrayRef<IREE::HAL::ExecutableTargetAttr> attrs)
      : targetAttrs(attrs) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override;

private:
  SmallVector<IREE::HAL::ExecutableTargetAttr, 4> targetAttrs;
};

FailureOr<MaterializeEncodingInfo>
materializeEncodingForTarget(RankedTensorType tensorType,
                             ExecutableTargetAttr targetAttr) {
  IREE::LinalgExt::EncodingAttr encoding =
      tensorType.getEncoding()
          .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (!encoding) {
    return failure();
  }
  // We only know about matmuls at the moment.
  auto user = encoding.getUser().getValue();
  if (user != EncodingUser::MATMUL && user != EncodingUser::BATCH_MATMUL) {
    return failure();
  }
  // Enumerate available tile shapes for the given encoding and target.
  auto elementTypes = llvm::to_vector(
      llvm::map_range(encoding.getElementTypes().getValue(), [](Attribute a) {
        return a.cast<TypeAttr>().getValue();
      }));
  SmallVector<TileMxNxK> enumeratedTileMxNxK =
      enumerateMatmulTileMxNxK(user, elementTypes, targetAttr);
  if (enumeratedTileMxNxK.empty()) {
    return failure();
  }
  // Check if the encoding specifies static narrow sizes for the M/N dimensions.
  // This can be used to choose a correspondingly narrow tile shape.
  // With microkernels, we keep this logic in sync with the set of actual
  // optimized microkernel tile functions to avoid a tile shape specialization
  // causing a fallback to a slow generic tile function. At the moment,
  // microkernel tile functions are only specialize for narrow M, not for narrow
  // N. Accordingly, we leave matmulNarrowN as 0 (default) when microkernels are
  // used. Generally it would be best to deal with narrow-N cases by transposing
  // the whole matmul and swapping LHS<->RHS, reducing the narrow-N case to
  // narrow-M.
  auto getIntOrZero = [](IntegerAttr a) {
    return a == IntegerAttr() ? 0 : a.getInt();
  };
  int64_t matmulNarrowM = getIntOrZero(encoding.getMatmulNarrow_M());
  int64_t matmulNarrowN = hasUkernel(targetAttr, "mmt4d")
                              ? 0
                              : getIntOrZero(encoding.getMatmulNarrow_N());
  // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
  // taking narrow dimensions into account.
  TileMxNxK chosenTileMxNxK =
      chooseMatmulTile(enumeratedTileMxNxK, matmulNarrowM, matmulNarrowN);
  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its role in the matmul.
  auto rank = tensorType.getRank();
  return getEncodingInfoForMatmul(encoding, rank, chosenTileMxNxK);
}

static MaterializeEncodingFn
getMaterializeEncodingFn(ExecutableTargetAttr targetAttr) {
  return
      [targetAttr](
          RankedTensorType tensorType) -> FailureOr<MaterializeEncodingInfo> {
        return materializeEncodingForTarget(tensorType, targetAttr);
      };
}

// Like getMaterializeEncodingFn, but iterating over an array of targets and
// returning the max of all tile sizes from each target, checking that other
// materialization info (permutations) agree.
//
// This is useful to compute padding amounts, in the materialization of
// UpperBoundTileSizeOp, in top-level functions that are not part of one HAL
// executable variant. There, the padding amounts only control the size of
// allocated buffers, so it's OK to over-estimate (only wasting some memory)
// but not under-estimate (would cause buffer overruns) padding amounts.
static MaterializeEncodingFn
getUpperBoundMaterializeEncodingFn(ArrayRef<ExecutableTargetAttr> targetAttrs) {
  return
      [targetAttrs](
          RankedTensorType tensorType) -> FailureOr<MaterializeEncodingInfo> {
        FailureOr<MaterializeEncodingInfo> result; // Defaults to failure.
        for (auto targetAttr : targetAttrs) {
          FailureOr<MaterializeEncodingInfo> info =
              materializeEncodingForTarget(tensorType, targetAttr);
          if (failed(info)) {
            // No info at this iteration. Ignore and continue.
            continue;
          }
          if (failed(result)) {
            // No preexisting result. Use this iteration's info and continue.
            result = info;
            continue;
          }
          // Merge this iteration's info into preexisting result info.
          // Check that permutations match, then record the max of tile sizes.
          if (info->innerDimsPos != result->innerDimsPos ||
              info->outerDimsPerm != result->outerDimsPerm) {
            return failure();
          }
          if (info->innerTileSizes.size() != result->innerTileSizes.size()) {
            return failure();
          }
          for (unsigned i = 0; i < info->innerTileSizes.size(); ++i) {
            if (ShapedType::isDynamic(info->innerTileSizes[i])) {
              result->innerTileSizes[i] = ShapedType::kDynamic;
            } else {
              result->innerTileSizes[i] =
                  std::max(result->innerTileSizes[i], info->innerTileSizes[i]);
            }
          }
        }
        return result;
      };
}

static FailureOr<MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc) {
  SmallVector<Type> resultTypes(tensorType.getRank(), builder.getIndexType());
  auto op = builder.create<IREE::Codegen::QueryTileSizesOp>(
      loc, resultTypes, TypeAttr::get(tensorType));
  MaterializeEncodingValueInfo result;
  result.innerTileSizes = op.getResults();
  return result;
}

static MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isVMVXBackend(targetAttr) && hasUkernel(targetAttr)) {
    return chooseDynamicEncodingInfoVMVXMicrokernels;
  }
  return {};
}

void CPUMaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  RewritePatternSet materializeEncodingPattern(context);
  if (!targetAttr)
    targetAttr = ExecutableTargetAttr::lookup(operation);
  auto materializeEncodingFn = getMaterializeEncodingFn(targetAttr);
  if (!materializeEncodingFn) {
    return signalPassFailure();
  }
  MaterializeEncodingTypeConverter typeConverter(materializeEncodingFn);
  MaterializeEncodingConversionTarget target(*context);
  auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
  populateMaterializeEncodingIntoPackUnPackPatterns(materializeEncodingPattern,
                                                    target, typeConverter,
                                                    materializeEncodingValueFn);

  if (failed(applyPartialConversion(operation, target,
                                    std::move(materializeEncodingPattern)))) {
    operation.emitOpError("materialization failed");
    return signalPassFailure();
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and resolve
  // dims ops.
  {
    RewritePatternSet patterns(context);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
      operation.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
}

void CPUMaterializeUpperBoundTileSizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  if (targetAttrs.empty()) {
    targetAttrs =
        IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(operation);
  }
  RewritePatternSet patterns(context);
  MaterializeEncodingFn materializeEncodingFn =
      getUpperBoundMaterializeEncodingFn(targetAttrs);
  if (!materializeEncodingFn) {
    return signalPassFailure();
  }
  populateMaterializeUpperBoundTileSizePatterns(patterns,
                                                materializeEncodingFn);
  if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
    operation.emitOpError(
        "encoding padding sizes materialization pattern failed");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createCPUMaterializeEncodingPass(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return std::make_unique<CPUMaterializeEncodingPass>(targetAttr);
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCPUMaterializeUpperBoundTileSizePass(
    ArrayRef<IREE::HAL::ExecutableTargetAttr> targetAttrs) {
  return std::make_unique<CPUMaterializeUpperBoundTileSizePass>(targetAttrs);
}

} // namespace mlir::iree_compiler
