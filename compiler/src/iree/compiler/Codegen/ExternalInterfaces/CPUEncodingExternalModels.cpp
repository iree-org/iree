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
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-cpu-encoding-external-models"

namespace mlir::iree_compiler::IREE::CPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

namespace {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static void transposeInPlace(MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // Not a vector case, so all three arrays in `info` have size at least 2,
  // outerDimsPerm may have size 3 if there is a batch dimension, but in all
  // cases, the last 2 entries of each array are M and N, not batch.
  auto transpose = [](SmallVector<int64_t> &a) {
    std::swap(a[a.size() - 2], a[a.size() - 1]);
  };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
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
      builder.create<tensor::EmptyOp>(loc, inputMixedSizes, outElemType);
  return builder
      .create<linalg::GenericOp>(
          loc, castedType, input, init, maps, iteratorTypes,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value castRes =
                b.create<arith::ExtUIOp>(nestedLoc, outElemType, args[0])
                    ->getResult(0);
            b.create<linalg::YieldOp>(nestedLoc, castRes);
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
        builder.create<tensor::ExpandShapeOp>(loc, newType, value, ri);
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
    LLVM_DEBUG(llvm::dbgs()
               << "candidate: "
               << llvm::interleaved(ArrayRef{tile.M, tile.N, tile.K})
               << " penalty:" << ratedTile.paddingPenalty << "\n");
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
  LLVM_DEBUG(
      llvm::dbgs() << "bestRatedTile: "
                   << llvm::interleaved(ArrayRef{
                          bestRatedTile.M, bestRatedTile.N, bestRatedTile.K})
                   << " penalty:" << bestRatedTile.paddingPenalty << "\n");
  return bestRatedTile;
}

FailureOr<Operation *> lowerContractionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
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
    return failure();
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return failure();
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
    result = builder.create<linalg::Mmt4DOp>(linalgOp.getLoc(), newResultType,
                                             ValueRange{newLhs, newRhs},
                                             ValueRange{newResult});
  } else {
    result = builder.create<linalg::BatchMmt4DOp>(
        linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
        ValueRange{newResult});
  }
  if (!ri.empty()) {
    result = builder.create<tensor::CollapseShapeOp>(
        linalgOp->getLoc(), operands[2].getType(), result->getResult(0), ri);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Interface methods implementaion for iree_cpu.cpu_encoding_layout.
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

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileArm64(TypeRange elementTypes,
                                                       DictionaryAttr config) {
  // Data-tiling for SVE is not implemented yet.
  if (hasFeature(config, "+sve") || hasFeature(config, "+sve2")) {
    return {};
  }

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
  return {};
}

struct CPUEncodingPackedLayoutMaterializerAttr
    : public PackedLayoutMaterializerAttrExternalModelBase<
          CPUEncodingPackedLayoutMaterializerAttr, CPUEncodingLayoutAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<CPUEncodingLayoutAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<CPUEncodingLayoutAttr>(attr);

    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());

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

    SmallVector<TileMxNxK> enumeratedTileMxNxK =
        enumerateCPUMatmulTiles(encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
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

struct CPUEncodingLayoutMaterializerAttr final
    : public EncodingLayoutMaterializerAttrExternalModelBase<
          CPUEncodingLayoutMaterializerAttr, CPUEncodingLayoutAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<CPUEncodingLayoutAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }

    FailureOr<Operation *> newOp = lowerContractionOpWithEncoding(
        b, linalgOp, convertedOperands,
        cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    return newOp.value_or(nullptr);
  }
};

struct CPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<CPULayoutResolverAttr,
                                                        CPUEncodingLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    storeNamedAttrIfPresent(configItems, config, "cpu_features");
    storeNamedAttrIfPresent(configItems, config, "target_triple");
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return CPUEncodingLayoutAttr::get(ctx,
                                      DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return CPUEncodingLayoutAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct CPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<CPUSerializableAttr,
                                                      CPUEncodingLayoutAttr> {

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

//===----------------------------------------------------------------------===//
// Interface methods implementaion for iree_cpu.vmvx_encoding_layout.
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
          VMVXEncodingPackedLayoutMaterializerAttr, VMVXEncodingLayoutAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<VMVXEncodingLayoutAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<VMVXEncodingLayoutAttr>(attr);

    auto encoding = llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
        type.getEncoding());

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
    // Choose a final matmul TileMxNxK from the above-enumarated tile shapes,
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

struct VMVXEncodingLayoutMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingLayoutMaterializerAttr, VMVXEncodingLayoutAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<VMVXEncodingLayoutAttr>(attr);
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }

    FailureOr<Operation *> newOp = lowerContractionOpWithEncoding(
        b, linalgOp, convertedOperands,
        cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    return newOp.value_or(nullptr);
  }
};

struct VMVXLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          VMVXLayoutResolverAttr, VMVXEncodingLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return VMVXEncodingLayoutAttr::get(ctx,
                                       DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return VMVXEncodingLayoutAttr::get(
        ctx, getPackedLayoutImpl(attr, type, /*addEncodingAttr=*/true));
  }
};

struct VMVXSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<VMVXSerializableAttr,
                                                      VMVXEncodingLayoutAttr> {
  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

} // namespace

void registerCPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::CPUEncodingLayoutAttr::attachInterface<
            CPUEncodingPackedLayoutMaterializerAttr,
            CPUEncodingLayoutMaterializerAttr, CPULayoutResolverAttr,
            CPUSerializableAttr>(*ctx);
        IREE::CPU::VMVXEncodingLayoutAttr::attachInterface<
            VMVXEncodingPackedLayoutMaterializerAttr,
            VMVXEncodingLayoutMaterializerAttr, VMVXLayoutResolverAttr,
            VMVXSerializableAttr>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
