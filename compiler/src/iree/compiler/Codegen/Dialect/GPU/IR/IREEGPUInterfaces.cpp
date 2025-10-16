// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/DialectImplementation.h"

#define DEBUG_TYPE "iree-gpu-interfaces"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.cpp.inc"

namespace mlir::iree_compiler::IREE::GPU {

using ::mlir::iree_compiler::IREE::Codegen::TileSwizzle;

//===----------------------------------------------------------------------===//
// DataTiledMMAInterfaceAttr
//===----------------------------------------------------------------------===//

/// Returns the swizzled tile distribution shape after applying the swizzle
/// permutation.
static SmallVector<int64_t>
getSwizzledDistributionShape(const TileSwizzle &swizzle) {
  SmallVector<int64_t> shape;
  for (TileSwizzle::ExpandShapeDimVectorType e : swizzle.expandShape) {
    for (TileSwizzle::Dim d : e) {
      shape.push_back(d.distributionSize);
    }
  }
  applyPermutationToVector(shape, swizzle.permutation);
  return shape;
}

void DataTiledMMAInterfaceAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &result) {
  SmallVector<Type> elementTypes;
  getElementTypes(elementTypes);
  for (auto [i, elementType] : llvm::enumerate(elementTypes)) {
    TileSwizzle swizzle = getTileSwizzle(i);
    SmallVector<int64_t> shape;
    for (TileSwizzle::ExpandShapeDimVectorType group : swizzle.expandShape) {
      for (TileSwizzle::Dim d : group) {
        shape.push_back(d.size);
      }
    }
    applyPermutationToVector(shape, swizzle.permutation);
    result.push_back(VectorType::get(shape, elementType));
  }
}

void DataTiledMMAInterfaceAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) {
  SmallVector<Type> elementTypes;
  getElementTypes(elementTypes);
  auto getShape = [=](unsigned operandIndex) {
    return sliceSwizzledShape(
        getTileSwizzle(operandIndex), [](TileSwizzle::Dim d) {
          return d.kind != TileSwizzle::Dim::Kind::CrossThread;
        });
  };
  for (auto [i, elementType] : llvm::enumerate(elementTypes)) {
    result.push_back(VectorType::get(getShape(i), elementType));
  }
}

std::optional<::mlir::SmallVector<int64_t, 2>>
DataTiledMMAInterfaceAttr::getUndistributedTileDimExpansion(
    int64_t operandIndex, int64_t logicalDim) {
  return std::nullopt;
}

LogicalResult DataTiledMMAInterfaceAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value threadId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) {
  TileSwizzle swizzle = getTileSwizzle(operandIndex);

  LDBG() << "DataTiledMMAInterfaceAttr::populateOperandOffsetsSizesStrides\n"
         << "    operand: " << operandIndex << "\n"
         << "    swizzle: " << swizzle << "\n";

  SmallVector<int64_t> distributionThreadSizes =
      getSwizzledDistributionShape(swizzle);

  // Obtain the offsets from delinearization along the distributionThreadSizes.
  // Use a delinearize without outer bound and throw away its initial result
  // to get clamping behavior.
  SmallVector<OpFoldResult> tileOffsets =
      affine::AffineDelinearizeIndexOp::create(
          builder, loc, getValueOrCreateConstantIndexOp(builder, loc, threadId),
          distributionThreadSizes, /*hasOuterBound=*/false)
          ->getResults()
          .drop_front();

  // For any dim that is distributed to more threads than the dim size, we need
  // to bound the offset to be within the dim bounds by dividing by the extra
  // distribution factor (see the definition of TileSwizzle::Dim).
  SmallVector<int64_t> layoutThreadSizes =
      sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind == TileSwizzle::Dim::Kind::CrossThread;
      });
  for (auto [offset, threadSize, distributionSize] : llvm::zip_equal(
           tileOffsets, layoutThreadSizes, distributionThreadSizes)) {
    if (distributionSize == threadSize) {
      continue;
    }
    Value distributionFactorVal = arith::ConstantIndexOp::create(
        builder, loc, llvm::divideCeil(distributionSize, threadSize));
    Value offsetVal = getValueOrCreateConstantIndexOp(builder, loc, offset);
    offset =
        arith::DivUIOp::create(builder, loc, offsetVal, distributionFactorVal)
            .getResult();
  }

  // Tile sizes are just the non-distributed dimensions, so anything that isn't
  // CrossThread.
  MLIRContext *ctx = builder.getContext();
  SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(
      ctx, sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind != TileSwizzle::Dim::Kind::CrossThread;
      }));
  // Strides are trivial: each slice is contiguous along the *expanded* dims
  // even if it may not be contiguous in the flattened layout.
  SmallVector<OpFoldResult> tileStrides(tileSizes.size(),
                                        builder.getIndexAttr(1));

  tileOffsets.assign(applyPermutation(tileOffsets, permutation));
  tileSizes.assign(applyPermutation(tileSizes, permutation));

  offsets.append(tileOffsets);
  sizes.append(tileSizes);
  strides.append(tileStrides);

  return success();
}

Attribute DataTiledMMAInterfaceAttr::getDistributionMappingKind() {
  return gpu::GPUThreadMappingAttr::get(getContext(),
                                        gpu::MappingId::LinearDim0);
}

OpFoldResult DataTiledMMAInterfaceAttr::getDistributionWorkerCount(
    OpBuilder &builder, Location loc, Operation *opToDistribute) {
  if (auto func = opToDistribute->getParentOfType<FunctionOpInterface>()) {
    if (std::optional<SmallVector<int64_t>> wgSizes = getWorkgroupSize(func)) {
      return getAsIndexOpFoldResult(getContext(),
                                    ShapedType::getNumElements(*wgSizes));
    }
  }
  return OpFoldResult();
}

} // namespace mlir::iree_compiler::IREE::GPU
