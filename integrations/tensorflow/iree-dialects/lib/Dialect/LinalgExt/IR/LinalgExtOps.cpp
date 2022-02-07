// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;
namespace IREE = mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}

Value IREE::LinalgExt::getDimValue(OpBuilder &builder, Location loc, Value v,
                                   int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

OpFoldResult IREE::LinalgExt::getDim(OpBuilder &builder, Location loc, Value v,
                                     int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyScatterOp(ScatterOp op) {
  if (op.inputs().size() != 2) {
    return op.emitOpError("expected two input operands");
  }
  if (op.outputs().size() != 1) {
    return op.emitOpError("expected one output operand");
  }
  auto checkDimensionsMatch = [&](ShapedType t1, ShapedType t2, unsigned dim) {
    return t1.getShape()[dim] == t2.getShape()[dim];
  };

  auto indicesType = op.getIndicesType();
  if (indicesType.getRank() != 2 ||
      !indicesType.getElementType().isInteger(32)) {
    return op.emitOpError(
        "expected indices to be of rank 2 of i32 element type");
  }
  auto indexDepth = op.getIndexDepth();
  if (indexDepth == ShapedType::kDynamicSize) {
    return op.emitOpError("expected index depth is static");
  }

  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto updateType = op.getUpdateType();
  if (updateType.getRank() < 1) {
    return op.emitOpError("expected update value to be at least rank 1");
  }
  if (!checkDimensionsMatch(indicesType, updateType, 0)) {
    return op.emitOpError(
        "mismatch in shape of indices and update value at dim#0");
  }
  auto originalType = op.getOriginalType();
  // indexDepth + update dims should match to original dims. The first dim of
  // update is the number of updates.
  if (originalType.getRank() != indexDepth + updateType.getRank() - 1) {
    return op.emitOpError(
        "mismatch in rank of update value, index depth and original value");
  }
  for (auto dim : llvm::seq<unsigned>(indexDepth, originalType.getRank())) {
    // Offset one because the first dim is the number of updates.
    if (updateType.getDimSize(1 + dim - indexDepth) !=
        originalType.getDimSize(dim)) {
      return op.emitOpError("mismatch in shape of update value dim#")
             << (1 + dim - indexDepth) << " and original value at dim#" << dim;
    }
  }
  Region &region = op.region();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op.emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!arg0Type.isIntOrFloat() || !arg1Type.isIntOrFloat()) {
    return op.emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op.emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return op.emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op.emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

SmallVector<StringRef> ScatterOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getUpdateType().getRank(),
                                       getParallelIteratorTypeName());
  return iteratorTypes;
}

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getUpdateType().getRank())) {
    Value ub = getDimValue(builder, loc, updates(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

Operation *ScatterOp::getTiledImplementation(OpBuilder &builder,
                                             ValueRange outputs,
                                             ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes,
                                             SmallVectorImpl<Value> &results) {
  assert(outputs.size() >= 1 && offsets.size() >= 1 && sizes.size() >= 1);
  Location loc = getLoc();
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  // Slice of the updates.
  auto updateRank = getUpdateType().getRank();
  SmallVector<OpFoldResult> updateStrides(updateRank, oneAttr);
  Value tiledUpdate =
      getSlice(builder, loc, updates(), offsets, sizes, updateStrides);
  assert(tiledUpdate && "failed to get slice of update");

  // Slice of indices.
  auto indicesRank = getIndicesType().getRank();
  SmallVector<OpFoldResult> indicesOffsets(indicesRank, zeroAttr);
  SmallVector<OpFoldResult> indicesSizes(indicesRank);
  indicesOffsets[0] = offsets[0];
  indicesSizes[0] = sizes[0];
  for (auto dim : llvm::seq<int64_t>(1, indicesRank)) {
    indicesSizes[dim] = getDim(builder, loc, indices(), dim);
  }
  SmallVector<OpFoldResult> indicesStrides(indicesRank, oneAttr);
  Value tiledIndices = getSlice(builder, loc, indices(), indicesOffsets,
                                indicesSizes, indicesStrides);
  assert(tiledIndices && "failed to get slice of indices");

  // Slice of the original.
  auto originalRank = getOriginalType().getRank();
  SmallVector<OpFoldResult> originalOffsets(originalRank, zeroAttr);
  SmallVector<OpFoldResult> originalSizes(originalRank);
  for (auto dim : llvm::seq<int64_t>(0, originalRank - updateRank + 1)) {
    originalSizes[dim] = getDim(builder, loc, original(), dim);
  }
  for (auto dim :
       llvm::seq<int64_t>(originalRank - updateRank + 1, originalRank)) {
    originalOffsets[dim] = offsets[dim - (originalRank - updateRank)];
    originalSizes[dim] = sizes[dim - (originalRank - updateRank)];
  }
  SmallVector<OpFoldResult> originalStrides(originalRank, oneAttr);
  Value tiledOriginal = getSlice(builder, loc, outputs[0], originalOffsets,
                                 originalSizes, originalStrides);
  assert(tiledOriginal && "failed to get slice of original tensor");

  SmallVector<Type> resultTypes;
  if (getNumResults()) {
    resultTypes.push_back(tiledOriginal.getType());
  }
  Operation *tiledScatterOp =
      cast<LinalgExtOp>(getOperation())
          .clone(builder, loc, resultTypes,
                 ValueRange{tiledUpdate, tiledIndices, tiledOriginal});
  for (auto result : llvm::enumerate(tiledScatterOp->getResults())) {
    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
        loc, result.value(), outputs[0], originalOffsets, originalSizes,
        originalStrides);
    results.push_back(insertSliceOp.getResult());
  }
  return tiledScatterOp;
}

LogicalResult ScatterOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  auto indexDepth = getIndexDepth();
  Value update = b.create<memref::LoadOp>(loc, updates(), ivs);
  SmallVector<Value> starts;
  SmallVector<Value> loadIndices;
  loadIndices.push_back(ivs.front());
  loadIndices.push_back(Value());
  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = b.create<arith::ConstantIndexOp>(loc, i);
    Value idx = b.create<memref::LoadOp>(loc, indices(), loadIndices);
    starts.push_back(b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx));
  }
  starts.append(std::next(ivs.begin()), ivs.end());
  Value init = b.create<memref::LoadOp>(loc, original(), starts);

  BlockAndValueMapping bvm;
  Block &block = region().front();
  bvm.map(block.getArgument(0), update);
  bvm.map(block.getArgument(1), init);
  for (auto &blockOp : block.without_terminator()) {
    b.clone(blockOp, bvm);
  }
  // The last op is linalg_ext.yield op. Store the operand to
  // destination.
  b.create<memref::StoreOp>(
      loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
      original(), starts);
  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

static LogicalResult verifySortOp(SortOp op) {
  if (op.getNumInputs()) {
    return op.emitOpError("does not expect to take any inputs");
  }
  if (op.getNumOutputs() == 0) {
    return op.emitOpError("expected at least one `outs` operand");
  }

  Block &block = op.region().front();
  size_t numOutputs = op.getNumOutputs();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op.emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int64_t rank = op.getOperandRank();
  ArrayRef<int64_t> shape = op.getOperandShape();
  if (rank > 1 && !op.dimensionAttr()) {
    return op.emitOpError("dimension must be specified if rank > 1");
  }
  int dimension = 0;
  if (op.dimensionAttr()) {
    dimension = op.dimension().getValue();
  }
  if (dimension < 0 || dimension >= rank) {
    return op.emitOpError("dimension must be within (0, ") << rank << "]";
  }

  for (auto indexedOperand : llvm::enumerate(op.outputs())) {
    int index = indexedOperand.index();
    auto operandType = op.getOperandType(index);
    if (operandType.getRank() != rank) {
      return op.emitOpError("expected operand ")
             << index << " to be rank " << rank << ", same as other operands";
    }
    if (operandType.getShape() != shape) {
      return op.emitOpError("expected operand ")
             << index << " to have same shape as other operands";
    }
    Type elemType = operandType.getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op.emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op.emitOpError("should yield exactly one operand");
  }
  auto ty = yieldOp.getOperand(0).getType().dyn_cast<IntegerType>();
  if (!ty || ty.getWidth() != 1) {
    return op.emitOpError("should yield i1 type");
  }

  return success();
}

SmallVector<StringRef> SortOp::getLoopIteratorTypes() {
  // All loops except the dimension to sort along are parallel.
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  iteratorTypes[getSortedDimension()] = getReductionIteratorTypeName();
  return iteratorTypes;
}

SmallVector<Range> SortOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = operand(0);
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<unsigned> SortOp::getPartitionableLoops(
    unsigned maxNumParallelDims) {
  auto range = llvm::seq<unsigned>(0, getOperandRank());
  SmallVector<unsigned> partitionableLoops(range.begin(), range.end());
  partitionableLoops.erase(
      std::next(partitionableLoops.begin(), getSortedDimension()));
  if (partitionableLoops.size() > maxNumParallelDims) {
    partitionableLoops.erase(
        partitionableLoops.begin(),
        std::next(partitionableLoops.begin(),
                  partitionableLoops.size() - maxNumParallelDims));
  }
  return partitionableLoops;
}

Operation *SortOp::getTiledImplementation(OpBuilder &builder,
                                          ValueRange outputs,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes,
                                          SmallVectorImpl<Value> &results) {
  assert(outputs.size() == this->outputs().size());
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  Location loc = getLoc();
  SmallVector<Value> tiledOperands(outputs.size());
  for (auto en : llvm::enumerate(outputs)) {
    tiledOperands[en.index()] =
        getSlice(builder, getLoc(), en.value(), offsets, sizes, strides);
    assert(tiledOperands[en.index()] && "failed to get slice of operand");
  }
  SmallVector<Type, 4> resultTypes;
  if (getNumResults()) {
    resultTypes = llvm::to_vector<4>(
        llvm::map_range(tiledOperands, [&](Value v) { return v.getType(); }));
  }
  Operation *tiledSortOp = cast<LinalgExtOp>(getOperation())
                               .clone(builder, loc, resultTypes, tiledOperands);
  for (auto result : llvm::enumerate(tiledSortOp->getResults())) {
    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
        loc, result.value(), outputs[result.index()], offsets, sizes, strides);
    results.push_back(insertSliceOp.getResult());
  }
  return tiledSortOp;
}

LogicalResult SortOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  auto sortDim = getSortedDimension();
  SmallVector<Value> indices, sortBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  // Bubble sort innermost loop.
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub;
  if (getOperandType(0).isDynamicDim(sortDim)) {
    ub = b.create<memref::DimOp>(loc, operand(0), sortDim);
  } else {
    ub = b.create<arith::ConstantIndexOp>(
        loc, getOperandType(0).getDimSize(sortDim));
  }
  ub = b.create<arith::SubIOp>(loc, ub, one);
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iters) {
        SmallVector<Value> indices(ivs);
        Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
        for (auto output : getOutputOperands()) {
          indices[sortDim] = iv;
          sortBlkArgs.push_back(
              b.create<memref::LoadOp>(loc, output->get(), indices));
          indices[sortDim] = ivPlusOne;
          sortBlkArgs.push_back(
              b.create<memref::LoadOp>(loc, output->get(), indices));
        }
      });

  auto &srcBlock = region().front();
  Region &region = scfFor.getRegion();
  BlockAndValueMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip(srcBlock.getArguments(), sortBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
  }
  Value cond = bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&region.front());
  b.create<scf::IfOp>(
      loc, TypeRange{}, cond,
      [&](OpBuilder &b, Location loc) {
        // Do not swap the pairs if true.
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        // Swap the pairs if false.
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value ivPlusOne =
            b.create<arith::AddIOp>(loc, scfFor.getInductionVar(), one);
        for (int i = 0, e = getNumOutputs(); i < e; ++i) {
          Value v1 = sortBlkArgs[i * 2];
          Value v2 = sortBlkArgs[i * 2 + 1];
          indices[sortDim] = scfFor.getInductionVar();
          b.create<memref::StoreOp>(loc, v2, getOutputOperand(i)->get(),
                                    indices);
          indices[sortDim] = ivPlusOne;
          b.create<memref::StoreOp>(loc, v1, getOutputOperand(i)->get(),
                                    indices);
        }
        b.create<scf::YieldOp>(loc);
      });
  b.create<scf::YieldOp>(loc);
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyFftOp(FftOp op) {
  auto length = op.getFftLength();
  // After tiling, it could be dynamic shape. (Because
  // subview/subtensor does not inference the type correctly
  // on (1 << x)) cases).
  if (length == ShapedType::kDynamicSize) return success();
  if (length & (length - 1)) {
    return op.emitOpError("only powers of 2 are handled currently");
  }
  if (!op.getNumInputs() || !op.isScalar(op.getInputOperand(0))) {
    return op.emitOpError("expected to carry `stage` input");
  }
  if (op.getNumInputs() != 1) {
    if (op.getNumInputs() != 3 || op.isScalar(op.getInputOperand(1)) ||
        op.isScalar(op.getInputOperand(2))) {
      return op.emitOpError("expected to carry real and imag coeff inputs");
    }
  }
  if (op.getNumOutputs() != 2) {
    return op.emitOpError("expected outputs to be real and imag tensor/memref");
  }
  return success();
}

SmallVector<StringRef> FftOp::getLoopIteratorTypes() {
  // There are `rank-1` outer loops. The fft itselfs has one loop for each
  // stage, which handles the merge step -- taking two half size tensors and
  // merge them into one tensor.
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  return iteratorTypes;
}

SmallVector<Range> FftOp::getIterationDomain(OpBuilder &builder) {
  SmallVector<Range> res;
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (auto en : llvm::enumerate(getOperandShape().drop_back())) {
    Value size;
    if (en.value() == ShapedType::kDynamicSize) {
      size = getDimValue(builder, loc, getReal(), en.index());
    } else {
      size = builder.create<arith::ConstantIndexOp>(loc, en.value());
    }
    res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/one});
  }

  Value size = getDimValue(builder, loc, getReal(), getOperandRank() - 1);
  Value stride = builder.create<arith::ShLIOp>(loc, one, getStage());
  res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/stride});
  return res;
}

void FftOp::generateScalarImplWithoutCoeffBuf(OpBuilder &b, Location loc,
                                              ArrayRef<Value> operands,
                                              Value wholeSize) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps(operands.size(), b.getMultiDimIdentityMap(rank));

  auto f32Type = b.getF32Type();
  auto indexToF32 = [](OpBuilder &builder, Location loc, Value v) -> Value {
    v = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), v);
    return builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), v);
  };

  // We will need exp(-2 * PI * j / m * I), compute "-2 * PI / m" for imag part
  // first.
  Value coeff = b.create<arith::ConstantFloatOp>(
      loc, llvm::APFloat(static_cast<float>(-2 * acos(-1))), f32Type);
  coeff = b.create<arith::DivFOp>(loc, coeff, indexToF32(b, loc, wholeSize));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{}, operands, maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value lhsReal = args[0];
        Value lhsImag = args[1];
        Value rhsReal = args[2];
        Value rhsImag = args[3];

        // Compute "-2 * PI / m * j"
        Value w = b.create<arith::MulFOp>(
            loc, coeff,
            indexToF32(b, loc, b.create<linalg::IndexOp>(loc, rank - 1)));
        Value wReal = b.create<math::CosOp>(loc, w);
        Value wImag = b.create<math::SinOp>(loc, w);

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

void FftOp::generateScalarImplWithCoeffBuf(OpBuilder &b, Location loc,
                                           ArrayRef<Value> operands) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps;
  // The size of coefficent buffer is epxected to match `2^(stage-1)`, which
  // equals to the last dim of operands.
  maps.append(
      2, AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
  maps.append(operands.size(), b.getMultiDimIdentityMap(rank));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{getRealCoeff(), getImagCoeff()}, operands,
      maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value wReal = args[0];
        Value wImag = args[1];
        Value lhsReal = args[2];
        Value lhsImag = args[3];
        Value rhsReal = args[4];
        Value rhsImag = args[5];

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

// Generates FFT stage scalar implementation. This follows Cooley–Tukey FFT
// algorithm. The pseudo reference code is:
//   let s <- stage of linalg_ext.fft
//   int m = 1 << s;
//   int mh = m >> 1;
//   for (int k = 0; k < n; k += m) {
//     for (int j = 0; j < mh; ++j) {
//       cplx w = exp(-2 * PI * j / m * I);
//       cplx t = w * a[k + j + mh];
//       cplx u = a[k + j];
//       a[k + j] = u + t;
//       a[k + j + mh] = u - t;
//     }
//   }
LogicalResult FftOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                  ValueRange ivs) {
  Value real = getReal();
  Value imag = getImag();
  Value stage = getStage();
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value wholeSize = b.create<arith::ShLIOp>(loc, one, stage);
  Value halfSize = b.create<arith::ShRSIOp>(loc, wholeSize, one);

  auto rank = getOperandRank();
  SmallVector<Value> operands;
  SmallVector<OpFoldResult> lhsIvs(ivs.begin(), ivs.end());
  SmallVector<OpFoldResult> ones(rank, b.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes(rank, b.getIndexAttr(1));
  sizes.back() = halfSize;
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, lhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, lhsIvs, sizes, ones));

  SmallVector<OpFoldResult> rhsIvs(ivs.begin(), ivs.end());
  rhsIvs.back() =
      b.create<arith::AddIOp>(loc, ivs.back(), halfSize).getResult();
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, rhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, rhsIvs, sizes, ones));

  if (hasCoeff()) {
    generateScalarImplWithCoeffBuf(b, loc, operands);
  } else {
    generateScalarImplWithoutCoeffBuf(b, loc, operands, wholeSize);
  }

  return success();
}

bool FftOp::payloadUsesValueFromOperand(OpOperand *) { return false; }

SmallVector<unsigned> FftOp::getPartitionableLoops(
    unsigned maxNumParallelDims) {
  auto range = llvm::seq<unsigned>(0, getOperandRank());
  SmallVector<unsigned> partitionableLoops(range.begin(), range.end());
  // Indices matter for coeff computation.
  if (!hasCoeff()) {
    partitionableLoops.pop_back();
  }
  if (partitionableLoops.size() > maxNumParallelDims) {
    partitionableLoops.erase(
        partitionableLoops.begin(),
        std::next(partitionableLoops.begin(),
                  partitionableLoops.size() - maxNumParallelDims));
  }
  return partitionableLoops;
}

Operation *FftOp::getTiledImplementation(OpBuilder &builder, ValueRange outputs,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         SmallVectorImpl<Value> &results) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();
  SmallVector<Value> tiledOperands(3);
  tiledOperands[0] = getStage();
  tiledOperands[1] = getRealCoeff();
  tiledOperands[2] = getImagCoeff();
  SmallVector<Type, 4> resultTypes;

  for (auto out : outputs) {
    tiledOperands.push_back(
        getSlice(builder, getLoc(), out, offsets, sizes, strides));
    if (hasTensorSemantics()) {
      resultTypes.push_back(tiledOperands.back().getType());
    }
  }
  Operation *tiledFftOp = cast<LinalgExtOp>(getOperation())
                              .clone(builder, loc, resultTypes, tiledOperands);
  for (auto result : llvm::enumerate(tiledFftOp->getResults())) {
    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
        loc, result.value(), outputs[result.index()], offsets, sizes, strides);
    results.push_back(insertSliceOp.getResult());
  }
  return tiledFftOp;
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyScanOp(ScanOp op) {
  if (op.getNumInputs() != 1) {
    return op.emitOpError("expected one input operands");
  }
  if (op.getNumOutputs() != 2) {
    return op.emitOpError("expected two output operands");
  }
  if (!op.input().getType().isa<ShapedType>()) {
    return op.emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = op.accumulator().getType().cast<ShapedType>();
  auto inputType = op.input().getType().cast<ShapedType>();
  auto outputType = op.output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op.emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op.emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != op.dimension()) expectedAccumulatorShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(llvm::zip(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op.emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op.emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op.emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op.emitOpError("incompatible input/output shapes");
  }
  return success();
}

SmallVector<Range> ScanOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<StringRef> ScanOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  iteratorTypes[dimension()] = getReductionIteratorTypeName();
  return iteratorTypes;
}

SmallVector<unsigned> ScanOp::getPartitionableLoops(
    unsigned maxNumParallelDims) {
  auto range = llvm::seq<unsigned>(0, getOperandRank());
  SmallVector<unsigned> partitionableLoops(range.begin(), range.end());
  partitionableLoops.erase(std::next(partitionableLoops.begin(), dimension()));
  return partitionableLoops;
}

// Generates naive scalar implementation of scan for a given operator f.
// For inclusive,
//     output[0] = input[0]
//     output[i] = f(output[i-1], input[i])
//
// For exclusive,
//     output[0] = 0
//     output[i] = f(output[i-1], input[i-1])

LogicalResult ScanOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  SmallVector<Value> indices, scanBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto scanDim = dimension();
  auto cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                      indices[scanDim], zero);
  bool isInclusive = inclusive();
  SmallVector<Value> accIndices;
  for (int i = 0; i < indices.size(); i++) {
    if (i != scanDim) accIndices.push_back(indices[i]);
  }

  auto scfIf = b.create<scf::IfOp>(
      loc, TypeRange{}, cond,
      [&](OpBuilder &b, Location loc) {
        if (isInclusive) {
          auto value = b.create<memref::LoadOp>(loc, input(), indices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        } else {
          auto value = b.create<memref::LoadOp>(loc, accumulator(), accIndices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        }
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value iv = indices[scanDim];
        Value ivMinusOne = b.create<arith::SubIOp>(loc, iv, one);
        indices[scanDim] = ivMinusOne;
        scanBlkArgs.push_back(b.create<memref::LoadOp>(loc, output(), indices));
        Value i0;
        if (!isInclusive) i0 = b.create<memref::LoadOp>(loc, input(), indices);
        indices[scanDim] = iv;
        if (isInclusive) i0 = b.create<memref::LoadOp>(loc, input(), indices);
        scanBlkArgs.push_back(i0);
      });

  auto &srcBlock = region().front();
  Region &region = scfIf.getElseRegion();
  BlockAndValueMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip(srcBlock.getArguments(), scanBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        output(), indices);
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        accumulator(), accIndices);
    b.create<scf::YieldOp>(loc);
  }
  return success();
}

Operation *ScanOp::getTiledImplementation(OpBuilder &builder,
                                          ValueRange outputs,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes,
                                          SmallVectorImpl<Value> &results) {
  assert(outputs.size() == this->outputs().size());
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), outputs[0], offsets, sizes, strides));
  SmallVector<OpFoldResult> accumOffsets, accumSizes, accumStrides;
  if (rank > 1) {
    for (int i = 0; i < rank; i++) {
      if (i != dimension()) {
        accumOffsets.push_back(offsets[i]);
        accumSizes.push_back(sizes[i]);
        accumStrides.push_back(strides[i]);
      }
    }
    tiledOperands.emplace_back(getSlice(
        builder, getLoc(), outputs[1], accumOffsets, accumSizes, accumStrides));
  } else {
    tiledOperands.emplace_back(outputs[1]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
  }

  Operation *tiledScanOp = cast<LinalgExtOp>(getOperation())
                               .clone(builder, loc, resultTypes, tiledOperands);
  for (auto result : llvm::enumerate(tiledScanOp->getResults())) {
    if ((result.index() == resultTypes.size() - 1) && (rank > 1)) {
      offsets = accumOffsets;
      sizes = accumSizes;
      strides = accumStrides;
    }
    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
        loc, result.value(), outputs[result.index()], offsets, sizes, strides);
    results.push_back(insertSliceOp.getResult());
  }
  return tiledScanOp;
}

static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

LogicalResult ScanOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyReverseOp(ReverseOp op) {
  if (op.getNumInputs() != 1) {
    return op.emitOpError("expected exactly one input");
  }
  if (op.getNumOutputs() != 1) {
    return op.emitOpError("expected exactly one output");
  }
  auto inputType = op.input().getType().cast<ShapedType>();
  auto outputType = op.output().getType().cast<ShapedType>();
  if (inputType.getElementType() != outputType.getElementType()) {
    return op.emitOpError(
        "expected input/output element types to be identical");
  }
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (inputShapes.size() != outputShapes.size()) {
    return op.emitOpError("expexted input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamicSize &&
                            std::get<1>(s) != ShapedType::kDynamicSize &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op.emitOpError("incompatible input/output shapes");
  }

  int64_t rank = op.getOperandRank();
  llvm::SmallSetVector<int64_t, 4> s;
  for (auto dim : op.dims()) {
    if (dim < 0 || dim >= rank) {
      return op.emitOpError("all the dimensions must be within [0, ")
             << rank << ")";
    }
    if (s.contains(dim)) {
      return op.emitOpError("expected dimensions numbers are all unique");
    }
    s.insert(dim);
  }

  return success();
}

bool ReverseOp::payloadUsesValueFromOperand(OpOperand *) { return false; }

SmallVector<StringRef> ReverseOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  return iteratorTypes;
}

SmallVector<Range> ReverseOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, input(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

LogicalResult ReverseOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  SmallVector<Value> mirrorIndices(ivs.begin(), ivs.end());
  for (auto dim : dims()) {
    auto size = getDimValue(b, loc, input(), dim);
    size = b.create<arith::SubIOp>(loc, size,
                                   b.create<arith::ConstantIndexOp>(loc, 1));
    mirrorIndices[dim] = b.create<arith::SubIOp>(loc, size, mirrorIndices[dim]);
  }
  Value val = b.create<memref::LoadOp>(loc, input(), ivs);
  b.create<memref::StoreOp>(loc, val, output(), mirrorIndices);
  return success();
}

Operation *ReverseOp::getTiledImplementation(OpBuilder &builder,
                                             ValueRange outputs,
                                             ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes,
                                             SmallVectorImpl<Value> &results) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, input(), offsets, sizes, strides));

  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  AffineMap map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/3, {sym0 - sym1 - sym2});
  SmallVector<OpFoldResult> mirrorOffsets(offsets.begin(), offsets.end());
  for (auto dim : dims()) {
    Value size = getDimValue(builder, loc, input(), dim);
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, mirrorOffsets[dim]);
    Value tileSize = getValueOrCreateConstantIndexOp(builder, loc, sizes[dim]);
    mirrorOffsets[dim] =
        builder
            .create<AffineApplyOp>(loc, map, ValueRange{size, offset, tileSize})
            .getResult();
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, output(), mirrorOffsets, sizes, strides));
    resultTypes.push_back(tiledOperands[1].getType());
  } else {
    tiledOperands.emplace_back(
        getSlice(builder, loc, output(), mirrorOffsets, sizes, strides));
  }

  Operation *tiledRevOp = cast<LinalgExtOp>(getOperation())
                              .clone(builder, loc, resultTypes, tiledOperands);

  for (auto result : llvm::enumerate(tiledRevOp->getResults())) {
    auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
        loc, result.value(), outputs[result.index()], mirrorOffsets, sizes,
        strides);
    results.push_back(insertSliceOp.getResult());
  }
  return tiledRevOp;
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                    \
  void OP_NAME::getEffects(                                               \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> \
          &effects) {                                                     \
    SmallVector<Value> inputBuffers = getInputBufferOperands();           \
    SmallVector<Value> outputBuffers = getOutputBufferOperands();         \
    getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,   \
                   outputBuffers);                                        \
  }

DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(SortOp)
DEFINE_OP_GET_EFFECTS(FftOp)
DEFINE_OP_GET_EFFECTS(ReverseOp)
DEFINE_OP_GET_EFFECTS(ScanOp)

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgExtOp> {
  using OpInterfaceRewritePattern<LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgExtOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
          if (opOperand->get().isa<BlockArgument>()) return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand) return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand *opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.source()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand *opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Clone op.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// LinalgExtDialect
//===----------------------------------------------------------------------===//

void IREELinalgExtDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<FoldTensorCastOp>(getContext());
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
