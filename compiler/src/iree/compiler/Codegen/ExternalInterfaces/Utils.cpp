// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE {
using IREE::Codegen::MaterializeEncodingInfo;

/// Computes the original index value for a given dimension after packing.
///
/// After packing, the original indices need to be computed from the packed
/// dimensions. For a dimension that is tiled:
///   original_index = outer_dim * tile_size + inner_dim
/// For a dimension that is not tiled:
///   original_index = outer_dim
///
/// The packed generic has identity output map, so dimensions are:
///   d0..d(outerRank-1): outer dimensions (permuted by outerDimsPerm)
///   d(outerRank)..d(outerRank+innerRank-1): inner dimensions
///
/// With swizzle, inner dimensions are further expanded and permuted.
static Value computePackedIndex(OpBuilder &builder, Location loc,
                                int64_t origDim, int64_t origRank,
                                const MaterializeEncodingInfo &encodingInfo,
                                ArrayRef<int64_t> outOffsetForDimsPos,
                                ArrayRef<int64_t> invOutSwizzlePerm) {
  ArrayRef<int64_t> outerDimsPerm = encodingInfo.outerDimsPerm;
  ArrayRef<int64_t> innerDimsPos = encodingInfo.innerDimsPos;
  ArrayRef<int64_t> innerTileSizes = encodingInfo.innerTileSizes;

  // Build inverse outer dims permutation for mapping original dims to packed
  // outer dims.
  SmallVector<int64_t> invOuterDimsPerm =
      invertPermutationVector(outerDimsPerm);

  // Find the outer packed dimension corresponding to this original dimension.
  int64_t outerPackedDim = invOuterDimsPerm[origDim];
  Value outerIdx =
      linalg::IndexOp::create(builder, loc, outerPackedDim).getResult();

  // Check if this dimension is tiled (in innerDimsPos).
  auto it = llvm::find(innerDimsPos, origDim);
  if (it == innerDimsPos.end()) {
    // Not tiled: original_index = outer_dim.
    return outerIdx;
  }

  // Tiled: original_index = outer_dim * tile_size + inner_dim.
  int64_t innerIdx = std::distance(innerDimsPos.begin(), it);
  int64_t tileSize = innerTileSizes[innerIdx];
  Value innerIdxVal;
  if (!encodingInfo.swizzle.has_value()) {
    // No swizzle: inner packed dim = outerRank + innerIdx.
    int64_t innerPackedDim = origRank + innerIdx;
    innerIdxVal =
        linalg::IndexOp::create(builder, loc, innerPackedDim).getResult();
  } else {
    // With swizzle: inner dimensions are expanded and permuted.
    // We need to compute the inner index from the expanded dimensions.
    const IREE::Codegen::TileSwizzle &swizzle = *encodingInfo.swizzle;
    ArrayRef<IREE::Codegen::TileSwizzle::Dim> expandDims =
        swizzle.expandShape[innerIdx];

    // Compute the starting offset for this inner tile's expanded dims.
    // The inner index is computed by combining the expanded dimensions
    // according to their sizes, but we need to account for the swizzle
    // permutation.
    int64_t expandedDimOffset = origRank + outOffsetForDimsPos[innerIdx];
    innerIdxVal = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    int64_t stride = 1;

    // Process expanded dimensions in reverse order to build the index.
    for (int64_t i = expandDims.size() - 1; i >= 0; --i) {
      int64_t expandedDim = expandedDimOffset + i;
      // Find where this expanded dimension ended up after swizzle.
      int64_t permutedDim = invOutSwizzlePerm[expandedDim];
      Value dimIdx =
          linalg::IndexOp::create(builder, loc, permutedDim).getResult();

      if (stride > 1) {
        Value strideVal =
            arith::ConstantIndexOp::create(builder, loc, stride).getResult();
        dimIdx =
            arith::MulIOp::create(builder, loc, dimIdx, strideVal).getResult();
      }
      innerIdxVal =
          arith::AddIOp::create(builder, loc, innerIdxVal, dimIdx).getResult();
      stride *= expandDims[i].size;
    }
  }

  // Compute outer_dim * tile_size + inner_dim.
  Value tileSizeVal =
      arith::ConstantIndexOp::create(builder, loc, tileSize).getResult();
  Value outerScaled =
      arith::MulIOp::create(builder, loc, outerIdx, tileSizeVal).getResult();
  return arith::AddIOp::create(builder, loc, outerScaled, innerIdxVal)
      .getResult();
}

void adjustTileSizesForBitcast(RankedTensorType type,
                               MaterializeEncodingInfo &info) {
  auto encoding =
      dyn_cast_if_present<Encoding::EncodingAttr>(type.getEncoding());
  if (!encoding) {
    return;
  }

  // Only adjust if the encoding has an original_element_type, indicating a
  // bitcast occurred.
  auto originalElementTypeAttr = encoding.getOriginalElementType();
  if (!originalElementTypeAttr) {
    return;
  }

  Type originalType = originalElementTypeAttr.getValue();
  Type storageType = type.getElementType();

  // No adjustment needed if types are the same.
  if (originalType == storageType) {
    return;
  }

  unsigned originalBits = originalType.getIntOrFloatBitWidth();
  unsigned storageBits = storageType.getIntOrFloatBitWidth();

  // One bit width must be a multiple of the other.
  assert((storageBits >= originalBits ? storageBits % originalBits
                                      : originalBits % storageBits) == 0 &&
         "bit widths must be multiples of each other");

  // Adjust the innermost tile size based on the bit width ratio between
  // original and storage types. The innermost dimension is where packing
  // occurs.
  if (info.innerTileSizes.empty()) {
    return;
  }
  int64_t &innermostTileSize = info.innerTileSizes.back();
  if (ShapedType::isDynamic(innermostTileSize)) {
    return;
  }

  // Scale the tile size: multiply by originalBits and divide by storageBits.
  // This scales down when storage > original (e.g., i4→i8 packing).
  int64_t scaledSize = innermostTileSize * originalBits;
  assert(scaledSize % storageBits == 0 &&
         "scaled tile size must be divisible by storage bits");
  innermostTileSize = scaledSize / storageBits;

  // Also adjust the swizzle's expandShape innermost dimension if present.
  // The expandShape describes how each inner tile dimension is further split,
  // and the innermost dimension's size must be scaled by the same ratio.
  if (info.swizzle && !info.swizzle->expandShape.empty()) {
    Codegen::TileSwizzle::ExpandShapeDimVectorType &innermostExpandDims =
        info.swizzle->expandShape.back();
    assert(!innermostExpandDims.empty() && "expand shape must be non-empty");
    Codegen::TileSwizzle::Dim &innermostDim = innermostExpandDims.back();
    int64_t scaledExpandSize = innermostDim.size * originalBits;
    assert(scaledExpandSize % storageBits == 0 &&
           "scaled expand size must be divisible by storage bits");
    innermostDim.size = scaledExpandSize / storageBits;
  }
}

Value calculatePackedStorageSizeInBytesImpl(Attribute attr, Location loc,
                                            OpBuilder &builder,
                                            RankedTensorType type,
                                            ValueRange dynamicDims) {
  auto deviceLayoutAttr =
      cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr);
  MaterializeEncodingInfo encodingInfo = deviceLayoutAttr.getEncodingInfo(type);
  SmallVector<int64_t> paddedShape(type.getShape());
  SmallVector<Value> paddedDynamicDims(dynamicDims);
  auto scalableFlags = encodingInfo.scalableTiles.value_or(
      Codegen::ScalableTileFlags(encodingInfo.innerTileSizes.size(), false));
  for (auto [dim, size, scalable] :
       llvm::zip_equal(encodingInfo.innerDimsPos, encodingInfo.innerTileSizes,
                       scalableFlags)) {
    // Only VMVX backend has dynamic inner tile sizes when ukernel is enabled.
    // It assumes that the padding size is 16. Ideally, the logic should be
    // moved to VMVX implementation details. However, we cook the logic here to
    // reduce code duplication.
    if (ShapedType::isDynamic(size)) {
      assert(isa<IREE::CPU::VMVXEncodingResolverAttr>(attr) &&
             "only VMVX backend attribute can handle dynamic tile sizes");
      size = 16;
    }

    // Host-side code does not support vscale ops yet - so we cannot create the
    // runtime SSA value properly as of now. #21317
    if (scalable) {
      size *= getUserVscaleValue();
    }
    // Do not create additional operations in the first place if the padding is
    // not needed.
    if (size == 1) {
      continue;
    }

    if (type.isDynamicDim(dim)) {
      dim = type.getDynamicDimIndex(dim);
      auto alignment = arith::ConstantIndexOp::create(builder, loc, size);
      paddedDynamicDims[dim] = arith::CeilDivSIOp::create(
          builder, loc, paddedDynamicDims[dim], alignment);
      paddedDynamicDims[dim] =
          arith::MulIOp::create(builder, loc, paddedDynamicDims[dim], alignment,
                                arith::IntegerOverflowFlags::nsw);
    } else {
      paddedShape[dim] = llvm::alignTo(paddedShape[dim], size);
    }
  }

  constexpr int64_t kNumBitsInByte = 8;
  int64_t numBytesPerElem = 1;
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits > kNumBitsInByte) {
    numBytesPerElem *= elementBits / kNumBitsInByte;
  }

  int64_t staticCount = numBytesPerElem;
  for (unsigned i = 0, e = type.getRank(); i < e; ++i) {
    if (!type.isDynamicDim(i)) {
      staticCount *= paddedShape[i];
    }
  }

  Value result =
      arith::ConstantIndexOp::create(builder, loc, staticCount).getResult();
  for (auto dim : paddedDynamicDims) {
    result = arith::MulIOp::create(builder, loc, result, dim,
                                   arith::IntegerOverflowFlags::nsw);
  }

  // Always pack the elements back-to-back for subtypes.
  if (elementBits < kNumBitsInByte) {
    if (kNumBitsInByte % elementBits) {
      assert(false && "unsupported subtype");
      return Value();
    }
    Value divisor = arith::ConstantIndexOp::create(
        builder, loc, kNumBitsInByte / elementBits);
    result = arith::CeilDivUIOp::create(builder, loc, result, divisor);
  }

  return result;
}

DictionaryAttr getPackedLayoutImpl(Attribute attr, RankedTensorType type,
                                   bool addEncodingAttr) {
  MLIRContext *ctx = attr.getContext();
  auto deviceLayoutAttr =
      cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr);
  const MaterializeEncodingInfo info = deviceLayoutAttr.getEncodingInfo(type);
  Attribute encodingInfoAttr =
      IREE::Codegen::serializeEncodingInfo(attr.getContext(), info);
  SmallVector<NamedAttribute> items;
  items.push_back(NamedAttribute(kEncodingInfoAttrName, encodingInfoAttr));
  auto encodingAttr = IREE::Encoding::getEncodingAttr(type);
  if (addEncodingAttr && encodingAttr) {
    items.push_back(NamedAttribute("encoding_attr", encodingAttr));
  }
  return DictionaryAttr::get(ctx, items);
}

void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name) {
  auto attr = dictAttr.getNamed(name);
  if (!attr) {
    return;
  }
  config.push_back(attr.value());
}

Operation *lowerFillOpWithResolvedLayouts(OpBuilder &builder,
                                          linalg::FillOp fillOp,
                                          TypeRange convertedResTypes,
                                          ValueRange convertedOperands) {
  return clone(builder, fillOp, convertedResTypes, convertedOperands);
}

Operation *lowerGenericOpWithResolvedLayouts(
    OpBuilder &builder, linalg::GenericOp genericOp,
    TypeRange convertedResTypes, ValueRange convertedOperands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!genericOp.hasPureTensorSemantics()) {
    return nullptr;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return nullptr;
  }

  ValueRange convertedInputOperands =
      convertedOperands.drop_back(convertedResTypes.size());
  ValueRange convertedOutputOperands =
      convertedOperands.take_back(convertedResTypes.size());
  if (genericOp.getNumResults() == 0) {
    return nullptr;
  }
  OpOperand *outputOperand = genericOp.getDpsInitOperand(0);
  AffineMap outputMap = genericOp.getMatchingIndexingMap(outputOperand);
  for (OpOperand &initOperand : genericOp.getDpsInitsMutable()) {
    if (genericOp.getMatchingIndexingMap(&initOperand) != outputMap) {
      return nullptr;
    }
  }
  // The pattern expects a generic op with an identity map for all outputs. If
  // this is not the case, then interchange the generic op before converting.
  if (!outputMap.isIdentity()) {
    if (!outputMap.isPermutation()) {
      return nullptr;
    }
    SmallVector<unsigned int> interchange = llvm::map_to_vector(
        outputMap.getResults(), [](AffineExpr expr) -> unsigned int {
          return cast<AffineDimExpr>(expr).getPosition();
        });
    // The method requires `RewriterBase` because it modifies ops in place.
    IRRewriter rewriter(builder);
    FailureOr<linalg::GenericOp> interchangedGenericOp =
        linalg::interchangeGenericOp(rewriter, genericOp, interchange);
    if (failed(interchangedGenericOp)) {
      return nullptr;
    }
    genericOp = interchangedGenericOp.value();
    outputOperand = genericOp.getDpsInitOperand(0);
    outputMap = genericOp.getMatchingIndexingMap(outputOperand);
  }
  // Step 1: Retrieve the output encoding materialization information and
  // compute the new indexing maps for the packed and potentially swizzled
  // layout. This consists of an outer dimension and inner dimension permutation
  // vectors for the packing and an expanded result dimension permutation vector
  // for the optional swizzling. This assumes that the output map is identity,
  // and that all iterator types are parallel.
  //
  // Running example:
  //
  // Given following output layout:
  //
  // outputType:              tensor<2x128x64xf32>
  // outputPackInfo:          innerDimsPos = [1, 2],
  //                          innerTileSizes = [128, 16]
  //                          outerDimsPerm = [0, 1, 2]
  // outputSwizzle:           expandShape = [[4, 8, 4], [4, 4]]
  //                          permutation = [1, 4, 0, 2, 3]}
  //
  // Retrieve and compute the permutation vectors for the packing outer and
  // inner dimension permutation and for the expanded swizzle permutation. Then,
  // calculate the permutation that would transform the swizzled output
  // dimension map into the identity dimension map. This is the inverse swizzle
  // permutation.
  //
  // outInverseOuterDimsPerm: [0, 1, 2]
  // outInnerDimsPos:         [1, 2]
  // outSwizzlePerm:          [0, 1, 2, 4, 7, 3, 5, 6]
  // invOutSwizzlePerm:       [0, 1, 2, 5, 3, 6, 7, 4]
  MaterializeEncodingInfo outMaterializeEncodingInfo =
      getEncodingInfoFromLayout(
          cast<RankedTensorType>(outputOperand->get().getType()), layoutAttr);
  if (IREE::Codegen::isIdentityLayout(outMaterializeEncodingInfo)) {
    return dropEncodingAndCloneOp(builder, genericOp.getOperation(),
                                  convertedInputOperands,
                                  convertedOutputOperands);
  }
  auto convertedResultType =
      cast<RankedTensorType>(convertedOutputOperands[0].getType());
  SmallVector<utils::IteratorType> iteratorTypes(convertedResultType.getRank(),
                                                 utils::IteratorType::parallel);

  SmallVector<int64_t> outInverseOuterDimsPerm =
      invertPermutationVector(outMaterializeEncodingInfo.outerDimsPerm);
  ArrayRef<int64_t> outInnerDimsPos = outMaterializeEncodingInfo.innerDimsPos;
  SmallVector<int64_t> outSwizzlePerm =
      llvm::to_vector(llvm::seq<int64_t>(0, convertedResultType.getRank()));
  if (outMaterializeEncodingInfo.swizzle.has_value()) {
    const int outRank =
        cast<RankedTensorType>(outputOperand->get().getType()).getRank();
    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, outRank));
    for (auto perm : outMaterializeEncodingInfo.swizzle->permutation) {
      transposePerm.push_back(outRank + perm);
    }
    applyPermutationToVector(outSwizzlePerm, transposePerm);
  }
  SmallVector<int64_t> invOutSwizzlePerm =
      invertPermutationVector(outSwizzlePerm);

  // Calculate the running offset for every dimension position for easy lookup
  // when calculating the packed result dimensions for every operand.
  // Example:
  //   expandShape == [[4, 8, 4], [4, 4]]
  // In this case:
  //   outOffsetForDimsPos == [0, 3]
  // So that whenever we need the real dimension for an entry (`outerIndex`,
  // `innerIndex`) in the 2D expanded shape vector, we can calculate it as:
  //   dim(outerIndex, innerIndex) = outOffsetForDimsPos[outerIndex] +
  //   innerIndex
  SmallVector<int64_t> outOffsetForDimsPos(outInnerDimsPos.size(), 0);
  if (outMaterializeEncodingInfo.swizzle.has_value()) {
    int64_t runningSize = 0;
    for (size_t i = 0; i < outInnerDimsPos.size(); i++) {
      outOffsetForDimsPos[i] = runningSize;
      runningSize += outMaterializeEncodingInfo.swizzle->expandShape[i].size();
    }
  }

  SmallVector<AffineMap> packedIndexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    AffineMap inputMap = genericOp.getMatchingIndexingMap(inputOperand);
    // Special case for 0D inputs. They will resolve to identity layout, so
    // skip the logic to compute the packed indexing map.
    if (inputMap.getNumResults() == 0) {
      auto packedInputMap = AffineMap::get(
          /*dimCount=*/iteratorTypes.size(), /*symbolCount=*/0, {},
          builder.getContext());
      packedIndexingMaps.push_back(packedInputMap);
      continue;
    }
    // Step 2: Retrieve the encoding for every input operand and perform the
    // outer dimension permutation, inner dimension expansion and permutation,
    // swizzle expansion and swizzle permutation.
    //
    // Running example:
    //
    // Given the input layout and indexing maps:
    //
    // inputType:       tensor<2x64xf32>
    // innerPackInfo:   innerDimsPos = [1]
    //                  innerTileSizes = [16]
    //                  outerDimsPerm = [0, 1]
    // innerSwizzle:    expandShape = [[4, 4]]
    //                  permutation = [1, 0]
    // inputMap:        [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
    //                   affine_map<(d0, d1, d2) -> (d0, d2)>]
    //
    // 1. Calculate the result dimensions from the indexing maps and perform the
    // outer dimension permutation:
    //
    // packedResultDims: [0, 2]
    //
    // 2. Perform inner dimension expansion, permutation and optional swizzle
    // expansion in one go. In this example, the inner dimension (64) would be
    // expanded into 4x16 based on `innerDimsPos` and `innerTileSizes` above,
    // and then expanded to 4x4x4 based on the swizzle.
    //
    // packedResultDims: [0, 2, 6, 7]
    //
    // 3. Perform the swizzle permutation:
    //
    // packedResultDims: [0, 2, 7, 6]
    MaterializeEncodingInfo materializeEncodingInfo = getEncodingInfoFromLayout(
        cast<RankedTensorType>(inputOperand->get().getType()), layoutAttr);
    if (IREE::Codegen::isIdentityLayout(materializeEncodingInfo)) {
      return nullptr;
    }
    ArrayRef<int64_t> innerDimsPos = materializeEncodingInfo.innerDimsPos;
    ArrayRef<int64_t> outerDimsPerm = materializeEncodingInfo.outerDimsPerm;
    // Permute result dims to the input packed domain, and map dims to the
    // output packed domain.
    SmallVector<int64_t> packedResultDims = llvm::map_to_vector(
        applyPermutation(inputMap.getResults(), outerDimsPerm),
        [&](AffineExpr expr) {
          auto dimExpr = cast<AffineDimExpr>(expr);
          return outInverseOuterDimsPerm[dimExpr.getPosition()];
        });
    // Add new dims for the inner tiles, taking the dim position from the
    // corresponding inner tile of the init operand.
    for (auto [idx, pos] : llvm::enumerate(innerDimsPos)) {
      auto dimPos = cast<AffineDimExpr>(inputMap.getResult(pos)).getPosition();
      for (auto [tileIdx, outDim] : llvm::enumerate(outInnerDimsPos)) {
        if (dimPos != outDim) {
          continue;
        }
        if (!materializeEncodingInfo.swizzle.has_value()) {
          packedResultDims.push_back(outputMap.getNumDims() + tileIdx);
          continue;
        }
        // In case of a layout with swizzle, an expanded set of dimensions
        // needs to be appended as specified by the swizzle's `expandedShape`
        // field. Note that the dimension index should be offset by the
        // calculated output starting offset as every dimension is now
        // transformed into an expanded sequence of indices and the correct
        // dimension index is:
        //   outOffsetForDimsPos[tileIdx] + innerIndex
        assert(idx < materializeEncodingInfo.swizzle->expandShape.size() &&
               "`innerDimsPos` index should not exceed the swizzle's "
               "`expandShape` size");
        const size_t dimSize =
            materializeEncodingInfo.swizzle->expandShape[idx].size();
        const int64_t outIdxOffset =
            outputMap.getNumDims() + outOffsetForDimsPos[tileIdx];
        for (size_t i = 0; i < dimSize; i++) {
          packedResultDims.push_back(outIdxOffset + i);
        }
      }
    }
    // In case of a layout with swizzle, the packed result dimensions need
    // to be transposed according to the swizzle's permutation vector.
    if (materializeEncodingInfo.swizzle.has_value()) {
      int inRank =
          cast<RankedTensorType>(inputOperand->get().getType()).getRank();
      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, inRank));
      for (auto perm : materializeEncodingInfo.swizzle->permutation) {
        transposePerm.push_back(inRank + perm);
      }
      applyPermutationToVector(packedResultDims, transposePerm);
    }

    // Step 3: Calculate the final packed result dimensions through the inverse
    // result dimensions permutation map. This effectively linearizes the packed
    // result dimensions with respect to the output dimensions. For example, if
    // the permuted output dimensions are [D0, D2, D1], this will transform all
    // packed operand result dimensions with the permutation map that would make
    // the output dimensions the identity map [D0, D1, D2], i.e. {D0 -> D0, D1
    // -> D2, D2 -> D1}. Suppose that the operand dimensions are [D0, D2], this
    // operation would transform it into [D0, D1] to align with the output
    // identity map.
    //
    // Running example:
    //
    // The packed and swizzled result dimensions for the input operand:
    //
    // packedResultDims:      [0, 2, 7, 6]
    //
    // Now we need to account for swizzled output result dimensions being
    // linearized to the identity map. This can be achieved by applying
    // `invOutSwizzlePerm` ([0, 1, 2, 5, 3, 6, 7, 4]):
    //
    // finalPackedResultDims: [0, 2, 4, 7]
    SmallVector<int64_t> finalPackedResultDims = llvm::map_to_vector(
        packedResultDims, [&](int64_t r) { return invOutSwizzlePerm[r]; });

    // Create the packed indexing map.
    SmallVector<AffineExpr> packedResultExprs =
        llvm::map_to_vector(finalPackedResultDims, [&](int64_t dim) {
          return builder.getAffineDimExpr(dim);
        });
    auto packedInputMap = AffineMap::get(
        /*dimCount=*/iteratorTypes.size(), /*symbolCount=*/0, packedResultExprs,
        builder.getContext());
    packedIndexingMaps.push_back(packedInputMap);
  }
  // Create the new packed identity map for the output.
  packedIndexingMaps.append(
      genericOp.getNumDpsInits(),
      builder.getMultiDimIdentityMap(convertedResultType.getRank()));
  SmallVector<Type> convertedResultTypes =
      llvm::map_to_vector(genericOp.getResultTypes(), [&](Type t) -> Type {
        return RankedTensorType::get(
            convertedResultType.getShape(),
            cast<RankedTensorType>(t).getElementType());
      });
  auto materializedGenericOp = linalg::GenericOp::create(
      builder, genericOp.getLoc(), convertedResultTypes, convertedInputOperands,
      convertedOutputOperands, packedIndexingMaps, iteratorTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));

  // Clone the region with proper remapping of linalg.index operations.
  // After packing, linalg.index ops need to compute the original index from
  // the packed dimensions.
  Block *origBlock = genericOp.getBlock();
  int64_t origRank =
      cast<RankedTensorType>(outputOperand->get().getType()).getRank();

  // Create the entry block for the new generic op with matching argument types.
  Region &newRegion = materializedGenericOp.getRegion();
  Block *newBlock = builder.createBlock(
      &newRegion, newRegion.begin(),
      llvm::to_vector(origBlock->getArgumentTypes()),
      SmallVector<Location>(origBlock->getNumArguments(), genericOp.getLoc()));

  IRMapping mapping;
  mapping.map(origBlock->getArguments(), newBlock->getArguments());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(newBlock);
  for (Operation &op : *origBlock) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(&op)) {
      Value newIndex = computePackedIndex(
          builder, indexOp.getLoc(), indexOp.getDim(), origRank,
          outMaterializeEncodingInfo, outOffsetForDimsPos, invOutSwizzlePerm);
      mapping.map(indexOp.getResult(), newIndex);
      continue;
    }
    builder.clone(op, mapping);
  }

  return materializedGenericOp.getOperation();
}

} // namespace mlir::iree_compiler::IREE
