// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {
//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

// Build a BarrierRegionOp with an empty.
void BarrierRegionOp::build(OpBuilder &b, OperationState &result,
                            TypeRange resultTypes, ValueRange inputs) {
  result.addOperands(inputs);
  (void)result.addRegion();
  result.addTypes(resultTypes);
  SmallVector<Location> blockArgLocs(inputs.size(), result.location);

  Region *region = result.regions[0].get();

  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), inputs.getTypes(), blockArgLocs);
}

LogicalResult BarrierRegionOp::verify() { return success(); }

LogicalResult BarrierRegionOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != getNumOperands()) {
    return emitError(
        "expected the block argument count to match operand count");
  }

  if (!llvm::all_of_zip(block.getArgumentTypes(), getOperandTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected block argument types to match operand types");
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = llvm::cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != getNumResults()) {
    return emitOpError(
        "expected body to yield same number of values as results");
  }

  if (!llvm::all_of_zip(yieldOp->getOperandTypes(), getResultTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected yielded value types to match result types");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

void ValueBarrierOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange input) {
  result.addOperands(input);
  result.addTypes(llvm::map_range(input, [](Value v) { return v.getType(); }));
}

LogicalResult ValueBarrierOp::verify() {
  if (getNumOperands() == 0) {
    return emitOpError("Atleast one input required");
  }

  // Make sure we either have all tensors or all vectors.
  if (hasTensorSemantics()) {
    bool allTensor =
        llvm::all_of(getInputTypes(), llvm::IsaPred<RankedTensorType>);
    if (!allTensor) {
      return emitOpError(
          "All inputs should be either of tensor or vector type");
    }
    return success();
  }

  bool allVector = llvm::all_of(getInputTypes(), llvm::IsaPred<VectorType>);
  if (!allVector) {
    return emitOpError("All inputs should be either of tensor or vector type");
  }

  return success();
}

// AMD Specific Operations

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

static RankedTensorType getMaximumStaticType(tensor::CastOp castOp) {
  auto inputType = dyn_cast<RankedTensorType>(castOp.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
  if (!inputType || !resultType) {
    return RankedTensorType();
  }

  assert(inputType.getRank() == resultType.getRank() &&
         "Rank must match for ranked -> ranked cast");

  SmallVector<int64_t> join;
  join.reserve(inputType.getRank());
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      join.push_back(resultType.getDimSize(i));
      continue;
    }
    if (resultType.isDynamicDim(i)) {
      join.push_back(inputType.getDimSize(i));
      continue;
    }

    // Cast verifier requires that static sizes match.
    join.push_back(inputType.getDimSize(i));
  }
  return RankedTensorType::get(join, inputType.getElementType());
}

struct FoldBufferCastOfTensorCast final
    : OpRewritePattern<BufferResourceCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferResourceCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the cast increases the amount of available static info.
    auto tensorCast = castOp.getInput().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }

    RankedTensorType maxStaticType = getMaximumStaticType(tensorCast);
    if (!maxStaticType || maxStaticType == castOp.getInput().getType()) {
      return failure();
    }

    Value newSource = tensorCast.getSource();
    if (newSource.getType() != maxStaticType) {
      // Cast to the type with maximum static information if the input and
      // result types contain different static info.
      newSource = tensor::CastOp::create(rewriter, castOp.getLoc(),
                                         maxStaticType, newSource);
    }
    auto newBufferCast = IREE::GPU::BufferResourceCastOp::create(
        rewriter, castOp.getLoc(), maxStaticType, newSource,
        castOp.getCacheSwizzleStride());
    newBufferCast->setDiscardableAttrs(castOp->getDiscardableAttrDictionary());

    // Cast back to the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        castOp, castOp.getResult().getType(), newBufferCast);
    return success();
  };
};

void BufferResourceCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *ctx) {
  results.add<FoldBufferCastOfTensorCast>(ctx);
}

//===----------------------------------------------------------------------===//
// CoalescedGatherDMAOp
//===----------------------------------------------------------------------===//

// DestinationStyleOpInterface implementation
MutableOperandRange CoalescedGatherDMAOp::getDpsInitsMutable() {
  return getInitMutable();
}

LogicalResult CoalescedGatherDMAOp::verify() {
  // Verify that this op must be inside a scf.forall op
  Operation *parent = (*this)->getParentOp();
  bool insideForall = false;
  while (parent) {
    if (isa<scf::ForallOp>(parent)) {
      insideForall = true;
      break;
    }
    parent = parent->getParentOp();
  }
  if (!insideForall) {
    return emitOpError("must be nested inside an scf.forall operation");
  }

  // Indices and source tensors have compatible types
  auto indicesType = cast<RankedTensorType>(getIndices().getType());
  auto initType = cast<RankedTensorType>(getInit().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  // Indices must be of i32 data type
  auto indicesElementType = indicesType.getElementType();
  if (!indicesElementType.isInteger(32)) {
    return emitOpError("indices must have i32 element type");
  }

  if (initType != resultType) {
    return emitOpError("init and result types must match");
  }

  // Size constraints between indices and init tensors
  // Calculate the number of elements in each tensor
  // Skip dynamic shape verification at compile time
  int64_t indicesNumElements = 1;
  for (int64_t dim : indicesType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      break;
    }
    indicesNumElements *= dim;
  }

  int64_t initNumElements = 1;
  for (int64_t dim : initType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      break;
    }
    initNumElements *= dim;
  }

  auto initElementType = initType.getElementType();
  unsigned initElementByteSize = initElementType.getIntOrFloatBitWidth() / 8;

  int64_t initTotalBytes = initNumElements * initElementByteSize;

  if (initTotalBytes % indicesNumElements != 0) {
    return emitOpError() << "the total byte size of init (" << initTotalBytes
                         << " = " << initNumElements << " elements * "
                         << initElementByteSize
                         << " bytes/element) must be divisible by "
                         << "the number of indices elements ("
                         << indicesNumElements << ")";
  }

  // The division result must be 1, 2, or 4
  int64_t ratio = initTotalBytes / indicesNumElements;
  if (ratio != 1 && ratio != 2 && ratio != 4) {
    return emitOpError() << "the ratio of init bytes to indices elements ("
                         << ratio << ") must be 1, 2, or 4";
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
