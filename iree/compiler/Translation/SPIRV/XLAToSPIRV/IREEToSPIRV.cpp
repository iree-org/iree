// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- IREEToSPIRV.cpp -----------------------------------------*- C++//-*-===//
//
// Translation of IREE statements in dispatch functions to SPIR-V.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/IREEToSPIRV.h"

namespace mlir {
namespace iree_compiler {

/// IREE::LoadInputOp is essentially a memcpy. Just update the `valueCache` with
/// the value of the operand.
LogicalResult IREELoadOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value> operands, TensorIndexToScalarValueMap &valueCache) const {
  auto loadOp = cast<IREE::LoadInputOp>(op);
  auto result = loadOp.getResult();
  valueCache.setValueAtIndex(result, index, operands[0]);
  return success();
}

/// IREE::StoreOp needs to write to the spv.globalVariable created for the
/// memref that holds the result of the dispatch function.
LogicalResult IREEStoreOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder,
    TensorIndexToScalarValueMap &valueCache) const {
  auto storeOp = cast<IREE::StoreOutputOp>(op);
  auto src = storeOp.src();
  SmallVector<AffineMap, 1> indices;
  index_computation_attribute::getIndexMapsForValue(src, indices);
  if (indices.size() != 1) {
    return storeOp.emitError(
        "expected to compute a single element of the tensor that is stored "
        "into the output memref");
  }
  auto var = valueCache.getBufferForArgument(storeOp.dst());
  if (!var) {
    return storeOp.emitError(
        "unable to find buffer that corresponds to the dst memref");
  }
  ArrayRef<int64_t> shape = {0};
  if (auto shapedType = src.getType().dyn_cast<ShapedType>()) {
    shape = shapedType.getShape();
  }
  auto ptr = genPointerOffset(builder, storeOp.getLoc(), valueCache, indices[0],
                              shape, var);
  auto scalarValue = valueCache.getValueAtIndex(src, indices[0]);
  builder.create<spirv::StoreOp>(storeOp.getLoc(), ptr, scalarValue,
                                 /*memory_access = */ nullptr,
                                 /*alignment = */ nullptr);
  return success();
}

/// IREE::ReturnOp in dispatch functions lowered to SPIR-V should have no
/// operands.
LogicalResult IREEReturnOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder,
    TensorIndexToScalarValueMap &valueCache) const {
  return success();
}

LogicalResult IREEStoreReduceOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder,
    TensorIndexToScalarValueMap &valueCache) const {
  auto storeReduceOp = cast<IREE::StoreReduceOp>(op);
  auto loc = storeReduceOp.getLoc();
  SmallVector<AffineMap, 1> srcIndices, dstIndices;
  Value src = storeReduceOp.src();
  index_computation_attribute::getIndexMapsForValue(src, srcIndices);
  if (srcIndices.size() != 1) {
    return storeReduceOp.emitError(
        "expected to compute a single element of the tensor that is reduced");
  }
  Value dst = storeReduceOp.dst();
  index_computation_attribute::getIndexMapsForValue(dst, dstIndices);
  if (dstIndices.size() != 1) {
    return storeReduceOp.emitError(
        "expected to write to a single element of the output of reduction");
  }
  auto srcValue = valueCache.getValueAtIndex(src, srcIndices[0]);
  auto dstBuffer = valueCache.getBufferForArgument(storeReduceOp.dst());
  auto ptr =
      genPointerOffset(builder, loc, valueCache, dstIndices[0],
                       dst.getType().cast<ShapedType>().getShape(), dstBuffer);
  builder.create<spirv::FunctionCallOp>(loc, ArrayRef<Type>(),
                                        storeReduceOp.reduction_fnAttr(),
                                        ArrayRef<Value>{srcValue, ptr});
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
