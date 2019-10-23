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
#include "iree/compiler/Translation/SPIRV/IREEToSPIRV.h"

namespace mlir {
namespace iree_compiler {

/// IREE::LoadInputOp is essentially a memcpy. Just update the `valueCache` with
/// the value of the operand.
LogicalResult IREELoadOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, ValueCache &valueCache) const {
  auto loadOp = cast<IREE::LoadInputOp>(op);
  auto result = loadOp.getResult();
  valueCache.setOperandDstValue(result, index, operands[0]);
  return success();
}

/// IREE::StoreOp needs to write to the spv.globalVariable created for the
/// memref that holds the result of the dispatch function.
LogicalResult IREEStoreOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache,
    DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
    ArrayRef<spirv::GlobalVariableOp> outputBuffers) const {
  auto storeOp = cast<IREE::StoreOutputOp>(op);
  auto src = storeOp.src();
  auto indices = affineExprCodegen.getIndices(src);
  if (indices.size() != 1) {
    return storeOp.emitError(
        "expected to compute a single element of the tensor that is stored "
        "into the output memref");
  }
  auto var = inputBuffers.lookup(storeOp.dst());
  if (!var) {
    return storeOp.emitError(
        "unable to find spv.globalVariable that corresponds to the dst memref");
  }
  auto ptr = genPointerOffset(builder, storeOp.getLoc(), affineExprCodegen,
                              indices[0], var);
  auto scalarValue = valueCache.getOperandDstValue(src, indices[0]);
  builder.create<spirv::StoreOp>(storeOp.getLoc(), ptr, scalarValue,
                                 /*memory_access = */ nullptr,
                                 /*alignment = */ nullptr);
  return success();
}

/// IREE::ReturnOp in dispatch functions lowered to SPIR-V should have no
/// operands.
LogicalResult IREEReturnOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache,
    DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
    ArrayRef<spirv::GlobalVariableOp> outputBuffers) const {
  builder.create<spirv::ReturnOp>(op->getLoc());
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
