// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LinalgBufferizePass.cpp - Pass to bufferize Linalg on tensors ------===//
//
// The overall bufferizarion algorithm is summarized here. Each of the
// individual steps are explained in detail later.
//
// Problem statement:
//
// The bufferization in this file is intended for converting tensor-operations
// into memref-operations for ops within a dispatch region. The goal is to reuse
// the buffers provided as inputs/outputs by the hal layer as memrefs for each
// of the operations. If the transformation cannot reuse input/output buffer to
// store an intermediate tensor, an allocation is done. This allocation is
// typically meant to be to target scratchspace memory.
//
// The algorithm has two phases an analysis phase and a tranformation phase.
//
// - The analysis phase walks the function and organizes relevant tensors
//   (tensors that need to be converted to memrefs) into equivalence clases. Two
//   tensors are part of the same equivalence class if they can eventually be
//   mapped to the same memref. This allows determining which operations can use
//   the buffer provided for the outputs to compute the results in place.
// - The transformation phase walks the function again and inserts corresponding
//   memref operations. The tensor operations are still kept around since the
//   analysis driving the transformation is based on the tensor values.
//   - Converting tensor operations to memref operations when all operands use
//     either buffers that are inputs to the dispatch or are allocated
//     temporarily within the dispatch region can be achieved by a
//     straight-forward walk.
//   - Reusing memref for the result of the dispatch for operations is more
//     involved and explained below.
//
//===----------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Bufferization helper functions using BlockAndValueMapping.
//===----------------------------------------------------------------------===//

/// Returns the dynamic dimensions of a Value `v` that is assumed to be
/// ShapedType.
static SmallVector<Value, 4> getDynamicDims(OpBuilder &b, Location loc,
                                            Value v) {
  SmallVector<Value, 4> dynamicDims;
  Type t = v.getType();
  for (auto shape : enumerate(t.cast<ShapedType>().getShape())) {
    if (shape.value() == ShapedType::kDynamicSize) {
      if (t.isa<MemRefType>()) {
        dynamicDims.push_back(
            b.createOrFold<memref::DimOp>(loc, v, shape.index()));
      } else {
        dynamicDims.push_back(
            b.createOrFold<tensor::DimOp>(loc, v, shape.index()));
      }
    }
  }
  return dynamicDims;
}

/// Allocates a memref for the results of an operation. Uses the
/// `InferShapedTypeOpInterface` where possible to get the shape of the output
/// in terms of the shapes of the operands.
static Value allocateBufferForResult(OpBuilder &b, Operation *op,
                                     unsigned resultNum,
                                     WorkgroupMemoryAllocationFn allocationFn) {
  assert(op->getNumResults() > resultNum);
  RankedTensorType resultType =
      op->getResult(resultNum).getType().cast<RankedTensorType>();
  SmallVector<Value, 4> dynamicDims;

  // Get the shape of the result
  Location loc = op->getLoc();
  if (auto shapedOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op)) {
    ReifiedRankedShapedTypeDims resultShape;
    if (failed(shapedOp.reifyResultShapes(b, resultShape))) {
      return nullptr;
    }
    for (auto shape : enumerate(resultShape[resultNum])) {
      if (resultType.isDynamicDim(shape.index())) {
        dynamicDims.push_back(shape.value());
      }
    }
  } else if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(op)) {
    dynamicDims = llvm::to_vector<4>(loadOp.sizes());
  } else if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    dynamicDims = llvm::to_vector<4>(sliceOp.sizes());
  } else if (auto subTensorInsertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    dynamicDims = getDynamicDims(b, loc, subTensorInsertOp.dest());
  } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
    dynamicDims = getDynamicDims(b, loc, transferWriteOp.source());
  } else {
    dynamicDims = getDynamicDims(b, loc, op->getResult(resultNum));
  }

  // If its a static allocation hoist it all the way up at begining of the
  // function.
  if (dynamicDims.empty()) {
    auto funcOp = op->getParentOfType<FuncOp>();
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&funcOp.front());
    return allocationFn(b, loc, resultType.getShape(),
                        resultType.getElementType(), dynamicDims);
  }
  return allocationFn(b, loc, resultType.getShape(),
                      resultType.getElementType(), dynamicDims);
}

template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         Attribute layout = Attribute(),
                                         Attribute memorySpace = nullptr) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

/// Checks if the offsets, sizes and strides with src, form a no-op
/// subview. This is true if
/// 1) The offsets are 0
/// 2) The strides are 1
/// 3) The sizes are same as that of the src.
/// For (3) when the shape is dynamic if the `src` is defined using an operation
/// that implements the `ShapeAwareOpInterface` (like
/// `hal.interface.binding.subspan`) then we can use that to check dynamic
/// equality.
/// Note: This could be written as a canonicalizer, but the subview formed
/// when there are dynamic shapes involved will have affine maps
/// that shouldnt be there. Resolving that is a pain. So dont generate the
/// subview to begin with.
static bool generatesNoOpSubView(Value src, ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes,
                                 ArrayRef<OpFoldResult> strides) {
  auto interfaceOp =
      dyn_cast_or_null<IREE::Util::ShapeAwareOpInterface>(src.getDefiningOp());
  if (!interfaceOp) {
    return false;
  }
  /// Check offsets are 0.
  if (llvm::any_of(offsets, [](OpFoldResult ofr) {
        Optional<int64_t> intValue = getConstantIntValue(ofr);
        return !intValue || intValue.getValue() != 0;
      })) {
    return false;
  }
  /// Check strides are 1.
  if (llvm::any_of(strides, [](OpFoldResult ofr) {
        Optional<int64_t> intValue = getConstantIntValue(ofr);
        return !intValue || intValue.getValue() != 1;
      })) {
    return false;
  }
  /// Check sizes are same as the source.
  auto dynamicDims = interfaceOp.getResultDynamicDims(0);
  unsigned dynamicDimsPos = 0;
  ArrayRef<int64_t> srcShape = src.getType().cast<MemRefType>().getShape();
  for (auto size : enumerate(sizes)) {
    if (Optional<int64_t> intValue = getConstantIntValue(size.value())) {
      if (intValue != srcShape[size.index()]) {
        return false;
      }
      continue;
    }
    if (dynamicDimsPos >= dynamicDims.size()) {
      return false;
    }
    if (size.value().get<Value>() == dynamicDims[dynamicDimsPos]) {
      dynamicDimsPos++;
      continue;
    }
    auto loadConstOp1 =
        size.value()
            .get<Value>()
            .getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>();
    auto loadConstOp2 =
        dynamicDims[dynamicDimsPos]
            .getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>();
    if (!loadConstOp1 || !loadConstOp2 ||
        loadConstOp1.index() != loadConstOp2.index()) {
      return false;
    }
    dynamicDimsPos++;
  }
  return true;
}

/// Creates a subview operation given the `src`, `offsets`, `sizes` and
/// `strides`. Handles the corner case where the `offsets`, `sizes` and
/// `strides` are empty in which case just forward the `src` value.  If the
/// `resultRank` does not match the source rank, uses a rank-reduced subview.
static Value createSubviewOp(OpBuilder &b, Location loc, unsigned resultRank,
                             Value src, ArrayRef<OpFoldResult> offsets,
                             ArrayRef<OpFoldResult> sizes,
                             ArrayRef<OpFoldResult> strides) {
  MemRefType srcType = src.getType().cast<MemRefType>();
  if (srcType.getRank() == resultRank &&
      generatesNoOpSubView(src, offsets, sizes, strides)) {
    return src;
  }
  MemRefType resultType;
  if (srcType.getRank() != resultRank) {
    resultType = memref::SubViewOp::inferRankReducedResultType(
                     resultRank, srcType, offsets, sizes, strides)
                     .cast<MemRefType>();
  }
  return b.create<memref::SubViewOp>(loc, resultType, src, offsets, sizes,
                                     strides);
}

//===----------------------------------------------------------------------===//
// There might be cases when the `value` stored into a
// `flow.dispatch.tensor.store` operation is obtained from operation that
// computes the value (say a `linalg` operation) through a series of `reshapes`,
// `cast` etc. When trying to reuse the buffer for the result passed in to the
// dispatch region for these operations, these operations need to be "replayed"
// in reverse so that the type of the buffer in the operation computing the
// value matches what is expected.
//
// For example,
// ```mlir
//   %buffer = hal.interface.binding.subspan .. : tensor<?xf32>
//   %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
//       outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
//   %value = tensor.collapse_shape %result [[0, 1]]
//       : tensor<?x?xf32> into tensor<?xf32>
//   flow.dispatch.tensor.store %value, %buffer[..] [..] [..]
// ```
//
// needs to be converted to
//
// ```mlir
//   %buffer = hal.interface.binding.subspan .. : memref<?xf32>
//   %result = subview %buffer[..] [..] [..] : memref<?xf32>
//   %value = linalg.reshape %result [affine_map<(d0, d1) -> (d0, d1)]
//       : memref<?xf32> into memref<?x?xf32>
//   linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
//       outs(%result : memref<?x?xf32>)
//   flow.dispatch.tensor.store %value, %buffer[..] [..] [..]
// ```
//
// ===----------------------------------------------------------------------===//

/// Returns the subview into the buffer that is supposed to be populated with
/// the `value` of the `flow.dispatch.tensor.store` operation. This can be used
/// to compute the results in place.
static Value getSubviewOpForTensorStoreOp(OpBuilder &b, Operation *storeOp,
                                          const BlockAndValueMapping &bvm) {
  SmallVector<Value, 4> operandsOfSubviewOp;
  auto op = cast<OffsetSizeAndStrideOpInterface>(storeOp);
  Value target, source;
  std::tie(source, target) =
      TypeSwitch<Operation *, std::tuple<Value, Value>>(op)
          .Case<IREE::Flow::DispatchTensorStoreOp>([&](auto storeOp) {
            return std::make_tuple(storeOp.value(), storeOp.target());
          })
          .Case<tensor::InsertSliceOp>([&](auto storeOp) {
            return std::make_tuple(storeOp.source(), storeOp.dest());
          })
          .Default([](Operation *) {
            return std::make_tuple<Value, Value>(nullptr, nullptr);
          });
  if (!target) return nullptr;

  // Clone the offset, size and stride values. They will be CSE-ed later.
  Operation *parentOp = storeOp->getParentOp();
  BlockAndValueMapping indexValMap;
  llvm::SetVector<Operation *> slice;
  auto cloneIndexValues = [&](ArrayRef<OpFoldResult> ofrs) {
    SmallVector<OpFoldResult> clonedVals;
    for (auto ofr : ofrs) {
      // Just copy the attributes.
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        clonedVals.push_back(attr);
        continue;
      }
      Value val = ofr.get<Value>();
      // If it is a block argument use the same value.
      if (val.isa<BlockArgument>()) {
        clonedVals.push_back(val);
        continue;
      }
      // The slice of ops needed for index computation need to be cloned to
      // avoid use-def violations. If the value has been cloned already, reuse
      // that.
      if (auto lookupVal = indexValMap.lookupOrNull(val)) {
        clonedVals.push_back(lookupVal);
        continue;
      }
      slice.clear();
      getBackwardSlice(val, &slice, [&](Operation *sliceOp) {
        return sliceOp->getParentOp() == parentOp;
      });
      for (auto sliceOp : slice) {
        if (!indexValMap.contains(sliceOp->getResult(0))) {
          b.clone(*sliceOp, indexValMap);
        }
      }
      if (Operation *definingOp = val.getDefiningOp()) {
        b.clone(*definingOp, indexValMap);
      }
      clonedVals.push_back(indexValMap.lookup(val));
    }
    return clonedVals;
  };
  SmallVector<OpFoldResult> subViewOffsets, subViewSizes, subViewStrides;
  subViewOffsets = cloneIndexValues(op.getMixedOffsets());
  subViewSizes = cloneIndexValues(op.getMixedSizes());
  subViewStrides = cloneIndexValues(op.getMixedStrides());
  Value subview = createSubviewOp(
      b, op.getLoc(), source.getType().cast<ShapedType>().getRank(),
      bvm.lookup(target), subViewOffsets, subViewSizes, subViewStrides);
  return subview;
}

/// Gets the reverse of a `tensor.collapse_shape` op to get a memref type that
/// can be used for in-place computation of the result of a dispatch region.
static Value getReverseOfReshapeOp(OpBuilder &b,
                                   tensor::CollapseShapeOp reshapeOp,
                                   Value resultBuffer) {
  auto memrefType = getMemrefTypeForTensor(
      reshapeOp.getSrcType(), {},
      resultBuffer.getType().cast<MemRefType>().getMemorySpace());
  return b.create<memref::ExpandShapeOp>(
      reshapeOp.getLoc(), memrefType, resultBuffer, reshapeOp.reassociation());
}

/// Gets the reverse of a `tensor.expand_shape` op to get a memref type that can
/// be used for in-place computation of the result of a dispatch region.
static Value getReverseOfReshapeOp(OpBuilder &b,
                                   tensor::ExpandShapeOp reshapeOp,
                                   Value resultBuffer) {
  return b.create<memref::CollapseShapeOp>(reshapeOp.getLoc(), resultBuffer,
                                           reshapeOp.getReassociationIndices());
}

/// Gets the reverse of a `tensor.cast` op to get a memref type that
/// can be used for in-place computation of the result of a disaptch region.
static Value getReverseOfCastOp(OpBuilder &b, tensor::CastOp castOp,
                                Value resultBuffer) {
  auto memrefType = getMemrefTypeForTensor(
      castOp.source().getType().cast<RankedTensorType>(),
      resultBuffer.getType().cast<MemRefType>().getLayout(),
      resultBuffer.getType().cast<MemRefType>().getMemorySpace());
  return b.create<memref::CastOp>(castOp.getLoc(), memrefType, resultBuffer);
}

/// Returns a tied result value give the operand. If no such result exists,
/// returns `nullptr`.
static Value getTiedResultForOperand(OpOperand &operand,
                                     const BufferizationPlan &plan) {
  for (Value result : operand.getOwner()->getResults()) {
    if (plan.isEquivalent(operand.get(), result)) {
      return result;
    }
  }
  return nullptr;
}

/// To perform updates directly into the result buffer, the uses need to be
/// walked to get to a value already mapped to a buffer or a
/// `flow.dispatch.tensor.store` operation. For each use, gets the tied result
/// and follow its uses. The traversed uses and thir tied results are returned
/// in `traversedUses`.
static Value walkUseToGetResultBuffer(
    OpBuilder &b, Value value, const BufferizationPlan &plan,
    const BlockAndValueMapping &bvm,
    SmallVectorImpl<std::pair<OpOperand *, Value>> &traversedUses) {
  Operation *op = value.getDefiningOp();
  if (!op) return nullptr;
  Operation *opParent = op->getParentOp();
  if (!opParent) return nullptr;
  while (value.hasOneUse()) {
    OpOperand &use = *value.use_begin();
    Operation *user = use.getOwner();
    bool isUserInSameScope = user->getParentOp() == opParent;
    if (isUserInSameScope &&
        isa<IREE::Flow::DispatchTensorStoreOp, tensor::InsertSliceOp>(user)) {
      return getSubviewOpForTensorStoreOp(b, user, bvm);
    }
    if (!isUserInSameScope && isa<scf::YieldOp>(user)) {
      value = cast<scf::ForOp>(user->getParentOp())
                  .getResult(use.getOperandNumber());
    } else {
      value = getTiedResultForOperand(use, plan);
    }
    if (!value) return nullptr;
    if (isUserInSameScope) {
      traversedUses.push_back(std::make_pair(&use, value));
    }
    if (auto resultBuffer = bvm.lookupOrNull(value)) return resultBuffer;
  }
  return nullptr;
}

/// For an operation whose `resultValue` is the result of the dispatch region,
/// gets the buffer to use to compute the value in-place.
static Value getInplaceResultBuffer(OpBuilder &b, OpResult resultValue,
                                    const BufferizationPlan &plan,
                                    BlockAndValueMapping &bvm) {
  // Traverse the use-def chains to get the `flow.dispatch.tensor.store`
  // operation keeping track of all the traversed operations. Note that the
  // equivalence set construction should ensure that all operations traversed
  // here have a single use.
  SmallVector<std::pair<OpOperand *, Value>> traversedUses;
  Value resultBuffer =
      walkUseToGetResultBuffer(b, resultValue, plan, bvm, traversedUses);
  if (!resultBuffer) return nullptr;

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "Pair :\n\tTensor :";
    resultValue.getOwner()->print(llvm::dbgs());
    llvm::dbgs() << "\nt\tMemref :";
    resultBuffer.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Now replay the instructions that are essentially doing type-conversion, in
  // reverse, to get the type needed for the operation computing the value.
  for (auto &it : llvm::reverse(traversedUses)) {
    Operation *op = it.first->getOwner();
    resultBuffer =
        TypeSwitch<Operation *, Value>(op)
            .Case<scf::IfOp, scf::ForOp, linalg::LinalgOp,
                  IREE::LinalgExt::LinalgExtOp, tensor::InsertSliceOp,
                  vector::TransferWriteOp>(
                [&](auto op) { return resultBuffer; })
            .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
                [&](auto reshapeOp) {
                  return getReverseOfReshapeOp(b, reshapeOp, resultBuffer);
                })
            .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
              return getReverseOfCastOp(b, castOp, resultBuffer);
            })
            .Default([&](Operation *) { return nullptr; });
    // TODO(ravishankarm): Maybe this needs to be an error. This should not have
    // happened.
    if (!resultBuffer) return nullptr;
    Value tiedResult = it.second;
    bvm.map(tiedResult, resultBuffer);
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Pair :\n\tTensor result "
                   << tiedResult.cast<OpResult>().getResultNumber() << " :";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\nt\tMemref :";
      resultBuffer.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
  return resultBuffer;
}

/// Converts a `tensor.cast` operation into a `memref.cast` operation with the
/// result aliasing the buffer for the operand.
static Value getAliasingBufferForResult(OpBuilder &b, tensor::CastOp castOp,
                                        BlockAndValueMapping &bvm) {
  Value inputBuffer = bvm.lookup(castOp.source());
  Value resultTensor = castOp.dest();
  auto outputType = getMemrefTypeForTensor(
      resultTensor.getType().cast<RankedTensorType>(), {},
      inputBuffer.getType().cast<MemRefType>().getMemorySpace());
  return b.create<memref::CastOp>(castOp.getLoc(), outputType, inputBuffer);
}

/// Returns the subview that indexes into the source of the interface buffer.
static Value getAliasingBufferForResult(OpBuilder &b,
                                        IREE::Flow::DispatchTensorLoadOp loadOp,
                                        BlockAndValueMapping &bvm) {
  Location loc = loadOp.getLoc();
  Value memref = bvm.lookup(loadOp.source());
  return createSubviewOp(b, loc,
                         loadOp.result().getType().cast<ShapedType>().getRank(),
                         memref, loadOp.getMixedOffsets(),
                         loadOp.getMixedSizes(), loadOp.getMixedStrides());
}

/// Converts a `tensor.collapse_shape` operation to a `memref.collapse_shape`
/// operation with the result aliasing the buffer for the operand.
static Value getAliasingBufferForReshapeResult(OpBuilder &b,
                                               tensor::CollapseShapeOp op,
                                               BlockAndValueMapping &bvm) {
  Location loc = op.getLoc();
  Value srcTensor = op.src();
  Value inputBuffer = bvm.lookup(srcTensor);

  // Create the reshape op.
  Value bufferReshape = b.create<memref::CollapseShapeOp>(
      loc, inputBuffer, op.getReassociationIndices());
  return bufferReshape;
}

/// Converts a `tensor.expand_shape` operation to a
/// `memref.expand_shape` operation with the result aliasing the buffer
/// for the operand.
static Value getAliasingBufferForReshapeResult(OpBuilder &b,
                                               tensor::ExpandShapeOp op,
                                               BlockAndValueMapping &bvm) {
  Location loc = op.getLoc();
  Value srcTensor = op.src();
  RankedTensorType resultTensorType = op.getResultType();
  Value inputBuffer = bvm.lookup(srcTensor);

  // Create the reshape op.
  MemRefType inputBufferType = inputBuffer.getType().cast<MemRefType>();
  auto reshapeResultType = getMemrefTypeForTensor(
      resultTensorType, {}, inputBufferType.getMemorySpace());
  Value bufferReshape = b.create<memref::ExpandShapeOp>(
      loc, reshapeResultType, inputBuffer, op.reassociation());
  return bufferReshape;
}

/// Converts a `subtensor` operation to a `subview` operation.
static Value getAliasingBufferForResult(OpBuilder &b, tensor::ExtractSliceOp op,
                                        BlockAndValueMapping &bvm) {
  Location loc = op.getLoc();
  Value srcTensor = op.source();
  Value inputBuffer = bvm.lookup(srcTensor);

  return createSubviewOp(b, loc, op.getType().getRank(), inputBuffer,
                         op.getMixedOffsets(), op.getMixedSizes(),
                         op.getMixedStrides());
}

/// Returns output buffers that aliases inputs.
static SmallVector<Value> getAliasingBuffersForResult(
    scf::ForOp scfFor, BlockAndValueMapping &bvm) {
  SmallVector<Value> aliasedBuffers(scfFor.getResults().size(), nullptr);
  for (int i = 0; i < scfFor.getResults().size(); ++i) {
    Value inputTensor = scfFor.getInitArgs()[i];
    if (!inputTensor.getType().isa<RankedTensorType>()) continue;
    Value inputBuffer = bvm.lookup(inputTensor);
    aliasedBuffers[i] = inputBuffer;
  }
  return aliasedBuffers;
}

/// Returns a `memref` for every result that aliases the buffer for one of its
/// operands. Returns the memref of the right shape/type based on the operation.
static SmallVector<Value, 4> getAliasingBuffersForResults(
    OpBuilder &b, Operation *op, BlockAndValueMapping &bvm) {
  return TypeSwitch<Operation *, SmallVector<Value, 4>>(op)
      .Case<IREE::Flow::DispatchTensorLoadOp, tensor::ExtractSliceOp,
            tensor::CastOp>([&](auto singleResultOp) -> SmallVector<Value, 4> {
        return {getAliasingBufferForResult(b, singleResultOp, bvm)};
      })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto reshapeOp) -> SmallVector<Value, 4> {
            return {getAliasingBufferForReshapeResult(b, reshapeOp, bvm)};
          })
      .Case<scf::ForOp>([&](auto scfFor) -> SmallVector<Value> {
        return getAliasingBuffersForResult(scfFor, bvm);
      })
      .Default([&](Operation *op) -> SmallVector<Value, 4> {
        return SmallVector<Value, 4>(op->getNumResults(), nullptr);
      });
}

/// For a result value, gets the operand that is tied with it. If no such
/// operand exists, returns `nullptr`.
static Value getTiedOperandForResult(OpResult result,
                                     const BufferizationPlan &plan) {
  for (Value operand : result.getOwner()->getOperands()) {
    if (plan.isEquivalent(operand, result)) {
      return operand;
    }
  }
  return nullptr;
}
static bool hasTiedOperandForResult(OpResult result,
                                    const BufferizationPlan &plan) {
  return static_cast<bool>(getTiedOperandForResult(result, plan));
}

/// Computes the `memrefs` to use for the result of an operation based on
/// - If the result has a tied operand reuse the buffer for the tied operand (or
///   an alias of it) as the buffer for the result. The `alaisingBuffer` vector
///   is expected to be as large as the number of results.
/// - If the result has no tied operands, the corresponding position in the
///   `aliasingBuffer` list must be `nullptr`.
/// - If the result is in the same equivalence set as the result of the dispatch
///   region (i.e. `value` operand of a `flow.dispatch.tensor.store`) then
///   return an alias/view of the buffer passed into the dispatch region to
///   store the results.
/// - Lastly, allocate a temporary buffer for the result using the passed
///   allocation function.
static LogicalResult getOrAllocateResultBuffers(
    OpBuilder &b, Operation *op, ArrayRef<Value> aliasingBuffers,
    BlockAndValueMapping &bvm, BufferizationPlan &plan,
    WorkgroupMemoryAllocationFn allocationFn) {
  assert(aliasingBuffers.size() == op->getNumResults());
  auto results = op->getResults();
  for (auto result : llvm::enumerate(results)) {
    if (!result.value().getType().isa<RankedTensorType>() ||
        bvm.contains(result.value())) {
      continue;
    }
    Value buffer;

    if (aliasingBuffers[result.index()] &&
        hasTiedOperandForResult(result.value(), plan)) {
      buffer = aliasingBuffers[result.index()];
    }
    if (!buffer && plan.isInStoreSet(result.value())) {
      buffer = getInplaceResultBuffer(b, result.value(), plan, bvm);
    }
    if (!buffer) {
      buffer = allocateBufferForResult(b, op, result.index(), allocationFn);
    }
    if (!buffer) {
      return op->emitError("unable to get result buffer for op");
    }
    bvm.map(result.value(), buffer);
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Pair :\n\tTensor result " << result.index() << ":";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\nt\tMemref :";
      buffer.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
  return success();
}

/// Convenience wrapper around core allocation function for the case where the
/// alias is the buffer for the result directly.
static LogicalResult getOrAllocateResultBuffers(
    OpBuilder &b, Operation *op, BlockAndValueMapping &bvm,
    BufferizationPlan &plan, WorkgroupMemoryAllocationFn allocationFn) {
  auto aliasingBuffers = llvm::to_vector<4>(
      llvm::map_range(op->getResults(), [&](OpResult result) {
        Value tiedOperand = getTiedOperandForResult(result, plan);
        return tiedOperand ? bvm.lookupOrNull(tiedOperand) : tiedOperand;
      }));
  return getOrAllocateResultBuffers(b, op, aliasingBuffers, bvm, plan,
                                    allocationFn);
}

/// Generic conversion pattern that matches any linalg::LinalgOp. This avoids
/// template instantiating one pattern for each linalg::LinalgOp. The method
/// expects all operands and results have already been mapped to memrefs.
template <typename OpTy>
static LogicalResult convertAnyLinalgOp(
    OpBuilder &b, OpTy op, BlockAndValueMapping &bvm, BufferizationPlan &plan,
    WorkgroupMemoryAllocationFn allocationFn) {
  // Skip linalg ops inserted by this pass.
  if (op.hasBufferSemantics()) return success();

  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    // For `linalg.poolin_*` ops, the input might be from a
    // `linalg.init_tensor`. In such cases, the `BlockAndValueMapping` wont have
    // a mapping for the buffer. Allocate a buffer for these.
    Value inputBuffer = bvm.lookupOrNull(opOperand->get());
    if (!inputBuffer) {
      OpResult definingOpResult = opOperand->get().dyn_cast<OpResult>();
      if (!definingOpResult) return failure();
      inputBuffer = allocateBufferForResult(b, definingOpResult.getOwner(),
                                            definingOpResult.getResultNumber(),
                                            allocationFn);
    }
    newInputBuffers.push_back(inputBuffer);
  }
  SmallVector<Value, 2> newOutputBuffers;
  auto results = op.getOperation()->getResults();
  auto outputs = op.getOutputOperands();
  for (auto it : llvm::zip(results, outputs)) {
    Value resultTensor = std::get<0>(it);
    Value resultBuffer = bvm.lookup(resultTensor);

    OpOperand *outOperand = std::get<1>(it);
    Value outTensor = outOperand->get();
    Value outBuffer = bvm.lookupOrNull(outTensor);
    if (outBuffer && !plan.isEquivalent(outTensor, resultTensor)) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
      if (!linalgOp || linalgOp.payloadUsesValueFromOperand(outOperand)) {
        createLinalgCopyOp(b, loc, outBuffer, resultBuffer);
      }
    }
    newOutputBuffers.push_back(resultBuffer);
  }

  SmallVector<Value, 8> newOperands(newInputBuffers.begin(),
                                    newInputBuffers.end());
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());
  op.clone(b, loc, {}, newOperands);
  return success();
}

/// Constants that return tensor types can be handled natively by the
/// backends. Here just provide a cast to memref to bridge the gap from tensors
/// to memrefs.
static LogicalResult convertConstantOp(OpBuilder &b,
                                       arith::ConstantOp constantOp,
                                       BlockAndValueMapping &bvm) {
  Value result = constantOp.getResult();
  assert(!bvm.lookupOrNull(result));
  RankedTensorType tensorType = result.getType().dyn_cast<RankedTensorType>();
  if (!tensorType) return success();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(constantOp);
  auto memrefType = getMemrefTypeForTensor(tensorType);
  Value memref = b.create<bufferization::ToMemrefOp>(constantOp.getLoc(),
                                                     memrefType, result);
  bvm.map(result, memref);
  return success();
}

/// Converts a `tensor.extract` operation into a `load`.
static LogicalResult convertTensorExtractOp(OpBuilder &b, tensor::ExtractOp op,
                                            const BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Value inputBuffer = bvm.lookup(op.tensor());
  Value load =
      b.createOrFold<memref::LoadOp>(op.getLoc(), inputBuffer, op.indices());
  // Since the value is the scalar, and `bvm` is used to only track tensor ->
  // memref mappings, just replace the uses directly.
  op.result().replaceAllUsesWith(load);
  return success();
}

/// Converts a `flow.dispatch.tensor.store` operation to memrefs. If the `value`
/// and `target` are in the same equivalent set, then there is nothing to do. If
/// no create a subview into the result buffer and copy the `value`.
static LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::Flow::DispatchTensorStoreOp storeOp,
    BlockAndValueMapping &bvm, BufferizationPlan &plan) {
  if (plan.isEquivalent(storeOp.target(), storeOp.value())) {
    return success();
  }
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);
  Value storeTo = bvm.lookup(storeOp.target());
  Value storeFrom = bvm.lookup(storeOp.value());

  Value subview = createSubviewOp(
      b, storeOp.getLoc(), storeFrom.getType().cast<ShapedType>().getRank(),
      storeTo, storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides());
  createLinalgCopyOp(b, storeOp->getLoc(), storeFrom, subview);
  return success();
}

/// Converts a `tensor.insert_slice` operation to buffers by
/// - Allocating a buffer for the result (if needed), and copying the
///   destination value into this buffer.
/// - Copying the source values into a subview of the result buffer.
static LogicalResult convertSubTensorInsertOp(OpBuilder &b,
                                              tensor::InsertSliceOp op,
                                              BlockAndValueMapping &bvm,
                                              BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.getResult();
  Value resultBuffer = bvm.lookup(result);

  // If `dest` and `result` are not equivalent, need a copy for that.
  Value dest = op.dest();
  if (!plan.isEquivalent(dest, result)) {
    Value destBuffer = bvm.lookup(dest);
    createLinalgCopyOp(b, loc, destBuffer, resultBuffer);
  }

  Value source = op.source();
  if (plan.isEquivalent(source, dest)) {
    return success();
  }

  // Copy from the source to the result subview.
  ShapedType sourceType = op.getSourceType();
  Value sourceBuffer = bvm.lookup(source);
  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  SmallVector<OpFoldResult> strides = op.getMixedStrides();
  Value subViewOp = createSubviewOp(b, loc, sourceType.getRank(), resultBuffer,
                                    offsets, sizes, strides);
  createLinalgCopyOp(b, loc, sourceBuffer, subViewOp);
  return success();
}

/// Converts a `tensor.insert` operations into a `memref.store`.
static LogicalResult convertTensorInsertOp(OpBuilder &b, tensor::InsertOp op,
                                           BlockAndValueMapping &bvm,
                                           BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.result();
  Value resultBuffer = bvm.lookup(result);
  if (!plan.isEquivalent(op.dest(), result)) {
    Value destBuffer = bvm.lookup(op.dest());
    createLinalgCopyOp(b, loc, destBuffer, resultBuffer);
  }

  b.create<memref::StoreOp>(loc, op.scalar(), resultBuffer, op.indices());
  return success();
}

/// Converts a vector.transfer_write op to use memref operands for source.
static LogicalResult convertVectorTransferWriteOp(OpBuilder &b,
                                                  vector::TransferWriteOp op,
                                                  BlockAndValueMapping &bvm,
                                                  BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.result();
  RankedTensorType resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!resultType) return success();
  Value resultBuffer = bvm.lookup(result);

  if (!plan.isEquivalent(op.source(), result) &&
      // If the source is linalg.init_tensor, then we don't care about the
      // initial value and can avoid the copy.
      !op.source().getDefiningOp<linalg::InitTensorOp>()) {
    Value destBuffer = bvm.lookup(op.source());
    createLinalgCopyOp(b, loc, destBuffer, resultBuffer);
  }

  // Create a new vector.transfer_write operation without a result value.
  b.create<vector::TransferWriteOp>(loc, op.vector(), resultBuffer,
                                    op.indices(), op.permutation_mapAttr(),
                                    op.mask(), op.in_boundsAttr());
  return success();
}

static LogicalResult convertScfForOp(OpBuilder &b, scf::ForOp forOp,
                                     BlockAndValueMapping &bvm,
                                     BufferizationPlan &plan) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Location loc = forOp.getLoc();
  for (auto arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!arg.value().getType().isa<RankedTensorType>()) continue;
    Value resultTensor = forOp.getResult(arg.index());
    Value resultBuffer = bvm.lookup(resultTensor);
    OpOperand &initOperand = forOp.getOpOperandForRegionIterArg(arg.value());
    Value yieldOperand = yieldOp.getOperand(arg.index());
    bvm.map(arg.value(), resultBuffer);
    bvm.map(yieldOperand, resultBuffer);
    if (!plan.isEquivalent(arg.value(), initOperand.get())) {
      Value initBuffer = bvm.lookup(initOperand.get());
      createLinalgCopyOp(b, loc, initBuffer, resultBuffer);
    }
  }
  return success();
}

static LogicalResult convertScfIfOp(OpBuilder &b, scf::IfOp ifOp,
                                    BlockAndValueMapping &bvm,
                                    BufferizationPlan &plan) {
  auto thenYieldOp = ifOp.thenYield();
  auto elseYieldOp = ifOp.elseYield();
  for (auto result : llvm::enumerate(ifOp.getResults())) {
    Value resultValue = result.value();
    if (!resultValue.getType().isa<RankedTensorType>()) continue;
    Value resultBuffer = bvm.lookup(resultValue);
    Value thenYield = thenYieldOp->getOperand(result.index());
    Value elseYield = elseYieldOp->getOperand(result.index());
    bvm.map(thenYield, resultBuffer);
    bvm.map(elseYield, resultBuffer);
  }
  return success();
}

/// If the alias of the buffer for an input oeprand cannot be used for the
/// "tied" results, need to do an explicit copy of the memory pointed to by the
/// aliased buffer into the buffer assigned to the result.
static void copyFromAliasingBufferToResultBuffer(
    OpBuilder &b, Location loc, ArrayRef<Value> tiedOperands,
    ArrayRef<Value> tiedResults, ArrayRef<Value> aliasingBuffers,
    BlockAndValueMapping &bvm, BufferizationPlan &plan) {
  for (auto result : enumerate(tiedResults)) {
    Value operand = tiedOperands[result.index()];
    if (!plan.isEquivalent(result.value(), operand)) {
      createLinalgCopyOp(b, loc, aliasingBuffers[result.index()],
                         bvm.lookup(result.value()));
    }
  }
}

/// Returns the static/dynamic mixed sizes of the memref.
static SmallVector<OpFoldResult> getMemrefSizes(OpBuilder &b, Location loc,
                                                Value memref) {
  auto inputShape = memref.getType().cast<ShapedType>().getShape();
  SmallVector<OpFoldResult> sizeMixedValues;
  for (int64_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] == ShapedType::kDynamicSize) {
      Value dim = b.create<memref::DimOp>(loc, memref, i);
      sizeMixedValues.push_back(dim);
    } else {
      sizeMixedValues.push_back(b.getI64IntegerAttr(inputShape[i]));
    }
  }
  return sizeMixedValues;
}

static LogicalResult convertPadTensorOp(OpBuilder &b, tensor::PadOp tensorPadOp,
                                        BlockAndValueMapping &bvm) {
  auto inputTensor = tensorPadOp.source();
  auto inputMemref = bvm.lookup(inputTensor);

  auto loc = tensorPadOp.getLoc();

  auto resultPaddedBuffer = bvm.lookup(tensorPadOp.result());

  // Get padding value and fill the result buffer.
  auto yeildOp = *tensorPadOp.region().getOps<tensor::YieldOp>().begin();
  Value paddingValue = yeildOp.value();

  auto constOp = paddingValue.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return tensorPadOp.emitError(
        "Converting linalg.pad_tensor with non-constant padding value");
  }
  if (constOp.getValue().isa<DenseElementsAttr>()) {
    return tensorPadOp.emitError(
        "Converting linalg.pad_tensor with non-scalar constant padding "
        "value");
  }

  b.create<linalg::FillOp>(loc, paddingValue, resultPaddedBuffer);

  // Get the interior region.
  SmallVector<OpFoldResult> sizeMixedValues =
      getMemrefSizes(b, loc, inputMemref);
  SmallVector<OpFoldResult> strides(
      inputMemref.getType().cast<ShapedType>().getRank(),
      b.getI64IntegerAttr(1));

  auto resultSubView = b.create<memref::SubViewOp>(loc, resultPaddedBuffer,
                                                   tensorPadOp.getMixedLowPad(),
                                                   sizeMixedValues, strides);
  // Copy to the interior region.
  createLinalgCopyOp(b, loc, inputMemref, resultSubView);
  return success();
}

namespace {
/// Pass to convert from tensor based ops to memref based ops.
class LinalgBufferizePass : public LinalgBufferizeBase<LinalgBufferizePass> {
 public:
  LinalgBufferizePass(WorkgroupMemoryAllocationFn fn) : allocationFn(fn) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect,
                    IREE::Util::UtilDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect, StandardOpsDialect,
                    mlir::math::MathDialect, mlir::arith::ArithmeticDialect>();
  }
  void runOnOperation() override;

 private:
  WorkgroupMemoryAllocationFn allocationFn;
};
}  // namespace

void LinalgBufferizePass::runOnOperation() {
  BufferizationPlan plan;
  FuncOp funcOp = getOperation();
  if (failed(createTensorEquivalenceClasses(funcOp, plan))) {
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();
  OpBuilder b(context);

  BlockAndValueMapping bvm;

  // First go over all hal.interface.binding.subspan ops and create counterparts
  // working with memrefs.
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    auto shapedType = subspanOp.getResult()
                          .getType()
                          .dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!shapedType || !shapedType.hasRank()) return;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(subspanOp);
    // Just change the result type of the InterfaceBindingSubspanOp to form
    // the base buffer.
    auto tensorType =
        subspanOp.result().getType().cast<IREE::Flow::DispatchTensorType>();
    auto memRefType = getMemrefTypeForTensor(tensorType);
    auto baseBuffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp->getLoc(), memRefType, subspanOp.set(), subspanOp.binding(),
        subspanOp.type(), subspanOp.byte_offset(), subspanOp.dynamic_dims(),
        subspanOp.alignmentAttr());
    auto alignment = baseBuffer.calculateAlignment();
    b.create<memref::AssumeAlignmentOp>(subspanOp->getLoc(), baseBuffer,
                                        alignment.value());
    bvm.map(subspanOp, baseBuffer);
  });

  // Visit all the operations that return `tensor`s and convert them to using
  // `memref`s.
  auto convertTensorProducingOps = [&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
          return convertConstantOp(b, constantOp, bvm);
        })
        .Case<IREE::Flow::DispatchTensorStoreOp>(
            [&](IREE::Flow::DispatchTensorStoreOp storeOp) {
              return convertInterfaceStoreTensorOp(b, storeOp, bvm, plan);
            })
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          if (failed(getOrAllocateResultBuffers(b, forOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertScfForOp(b, forOp, bvm, plan);
        })
        .Case<scf::IfOp>([&](scf::IfOp ifOp) {
          if (failed(getOrAllocateResultBuffers(b, ifOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertScfIfOp(b, ifOp, bvm, plan);
        })
        .Case<IREE::Flow::DispatchTensorLoadOp, tensor::CollapseShapeOp,
              tensor::ExpandShapeOp, tensor::ExtractSliceOp, tensor::CastOp>(
            [&](auto aliasingOp) {
              auto aliasingBuffers =
                  getAliasingBuffersForResults(b, aliasingOp, bvm);
              if (failed(getOrAllocateResultBuffers(b, aliasingOp,
                                                    aliasingBuffers, bvm, plan,
                                                    allocationFn))) {
                return failure();
              }
              copyFromAliasingBufferToResultBuffer(
                  b, aliasingOp->getLoc(), aliasingOp->getOperand(0),
                  aliasingOp->getResult(0), aliasingBuffers, bvm, plan);
              return success();
            })
        .Case<tensor::PadOp>([&](tensor::PadOp tensorPadOp) {
          if (failed(getOrAllocateResultBuffers(b, tensorPadOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertPadTensorOp(b, tensorPadOp, bvm);
        })
        .Case<linalg::LinalgOp, IREE::LinalgExt::LinalgExtOp>([&](auto op) {
          if (failed(
                  getOrAllocateResultBuffers(b, op, bvm, plan, allocationFn))) {
            return failure();
          }
          return convertAnyLinalgOp(b, op, bvm, plan, allocationFn);
        })
        .Case<tensor::InsertSliceOp>(
            [&](tensor::InsertSliceOp subTensorInsertOp) {
              if (failed(getOrAllocateResultBuffers(b, subTensorInsertOp, bvm,
                                                    plan, allocationFn))) {
                return failure();
              }
              return convertSubTensorInsertOp(b, subTensorInsertOp, bvm, plan);
            })
        .Case<tensor::InsertOp>([&](tensor::InsertOp tensorInsertOp) {
          if (failed(getOrAllocateResultBuffers(b, tensorInsertOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertTensorInsertOp(b, tensorInsertOp, bvm, plan);
        })
        .Case<vector::TransferWriteOp>(
            [&](vector::TransferWriteOp transferWriteOp) {
              if (failed(getOrAllocateResultBuffers(b, transferWriteOp, bvm,
                                                    plan, allocationFn))) {
                return failure();
              }
              return convertVectorTransferWriteOp(b, transferWriteOp, bvm,
                                                  plan);
            })
        .Default([&](Operation *op) { return success(); });
  };
  auto walkResult =
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        b.setInsertionPoint(op);
        return convertTensorProducingOps(op);
      });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  // Lastly visit the non-tensor return operations that still use `tensor`
  // values. These need to be updated to use the corresponding `memref` values,
  // but dont need to update the block-and-value mapping.
  auto convertNonTensorProducingOps = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<tensor::ExtractOp>([&](tensor::ExtractOp op) {
          return convertTensorExtractOp(b, op, bvm);
        })
        .Case<vector::TransferReadOp>([&](auto op) {
          for (unsigned i : llvm::seq<unsigned>(0, op->getNumOperands())) {
            Value operand = op->getOperand(i);
            if (operand.getType().isa<RankedTensorType>()) {
              Value remappedVal = bvm.lookupOrNull(operand);
              if (remappedVal) op->setOperand(i, remappedVal);
            }
          }
          return success();
        })
        .Case<tensor::DimOp>([&](tensor::DimOp dimOp) {
          Value operand = dimOp.source();
          Value remappedVal = bvm.lookupOrNull(operand);
          if (remappedVal) {
            Value newDimOp = b.create<memref::DimOp>(
                dimOp.getLoc(), remappedVal, dimOp.index());
            dimOp.replaceAllUsesWith(newDimOp);
          }
          return success();
        })
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          // To canonicalize the `scf.for` tensor result/operand/yield value
          // away, forward the init argument to the yeild of the loop.
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          for (auto arg : llvm::enumerate(forOp.getIterOperands())) {
            if (!arg.value().getType().isa<RankedTensorType>()) continue;
            yieldOp.setOperand(arg.index(), arg.value());
          }
          return success();
        })
        .Case<IREE::Flow::DispatchTensorStoreOp>([&](auto op) {
          op.erase();
          return success();
        })
        .Default([&](Operation *op) { return success(); });
  };

  walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    b.setInsertionPoint(op);
    return convertNonTensorProducingOps(op);
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
}

static Value defaultAllocationFn(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> staticShape,
                                 Type elementType,
                                 ArrayRef<Value> dynamicSizes) {
  auto allocationType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes);
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn) {
  return std::make_unique<LinalgBufferizePass>(
      allocationFn ? allocationFn : defaultAllocationFn);
}

void addLinalgBufferizePasses(OpPassManager &passManager,
                              WorkgroupMemoryAllocationFn allocationFn) {
  passManager.addNestedPass<FuncOp>(createLinalgBufferizePass(allocationFn));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
  // passManager.addPass(createBufferHoistingPass());
  // TODO(nicolasvasilache): bug in buffer loop hoisting with
  // dynamic_linalg_matmul_on_tensors_fuse_0.mlir
  // passManager.addPass(createBufferLoopHoistingPass());
}

}  // namespace iree_compiler
}  // namespace mlir
