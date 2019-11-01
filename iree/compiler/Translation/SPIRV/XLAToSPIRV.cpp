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

//===- XLAToSPIRV.cpp ------------------------------------------*- C++//-*-===//
//
// Implementation of SPIR-V Code-generation for xla_hlo operations within IREE
// Dispatch functions
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult XLAConcatenateOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache) const {
  auto concatenateOp = cast<xla_hlo::ConcatenateOp>(op);
  auto loc = concatenateOp.getLoc();
  auto i32Type = builder.getIntegerType(32);
  auto i1Type = builder.getI1Type();
  int append_dim = concatenateOp.dimension().getZExtValue();
  auto dimIndex = affineExprCodegen.getValue(index.getResult(append_dim),
                                             builder.saveInsertionPoint(), loc);

  int offset = op->getOperand(0)
                   ->getType()
                   .cast<RankedTensorType>()
                   .getShape()[append_dim];
  Value *resultVal = operands[0];
  for (auto operandIt : llvm::enumerate(op->getOperands())) {
    // The first operand is already saved in resultVal.
    if (operandIt.index() == 0) continue;

    // Only select values that offset <= d < offset + operand_shape[append_dim].
    // Since later values will be replaced in the later iterations, only check
    // d >= offset here.
    Value *cond = builder.create<spirv::ConstantOp>(loc, i1Type,
                                                    builder.getBoolAttr(true));
    auto offsetVar = builder.create<spirv::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(offset));
    auto checkLb = builder.create<spirv::SGreaterThanEqualOp>(
        loc, i1Type, dimIndex, offsetVar);
    cond = builder.create<spirv::LogicalAndOp>(loc, i1Type, cond, checkLb);
    resultVal = builder.create<spirv::SelectOp>(
        loc, cond, operands[operandIt.index()], resultVal);
    auto operandShape =
        operandIt.value()->getType().cast<RankedTensorType>().getShape();
    offset += operandShape[append_dim];
  }
  valueCache.setOperandDstValue(op->getResult(0), index, resultVal);
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult XLAPadOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache) const {
  auto padOp = cast<xla_hlo::PadOp>(op);
  const auto &edgePaddingLow = padOp.edge_padding_low();
  const auto &interiorPadding = padOp.interior_padding();
  const auto &edgePaddingHigh = padOp.edge_padding_high();
  // If the `index` of the result at a particular dimension i, is d_i, check if
  //
  // (d_i >= edge_padding_low[i]) &&
  // ((d_i - edge_padding_low[i]) % (interior_padding[i]+1) == 0) &&
  // (d_i < (edge_padding_low[i] + (interior_padding[i]+1) * operand_shape[i])).
  //
  // If true, then use the value of the operand, or use the
  // padding value.
  auto loc = padOp.getLoc();
  auto i32Type = builder.getIntegerType(32);
  auto i1Type = builder.getI1Type();
  auto zero = builder.create<spirv::ConstantOp>(loc, i32Type,
                                                builder.getI32IntegerAttr(0));
  Value *cond = builder.create<spirv::ConstantOp>(loc, builder.getI1Type(),
                                                  builder.getBoolAttr(true));
  auto operandType = padOp.operand()->getType().cast<RankedTensorType>();
  if (!operandType.hasStaticShape()) {
    return padOp.emitError("pad op codegen supported only for static shapes");
  }
  auto operandShape = operandType.getShape();
  for (auto resultIndex : enumerate(index.getResults())) {
    auto i = resultIndex.index();

    // (edge_padding_low[i] + (interior_padding[i]+1) * operand_shape[i])
    int64_t paddingLow = edgePaddingLow.getValue<IntegerAttr>(i).getInt();
    int64_t paddingStride =
        interiorPadding.getValue<IntegerAttr>(i).getInt() + 1;
    int64_t paddingHigh = edgePaddingHigh.getValue<IntegerAttr>(i).getInt();
    if (paddingLow == 0 && paddingStride == 1 && paddingHigh == 0) {
      continue;
    }
    auto edgePadding = builder.create<spirv::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(paddingLow));
    auto stride = builder.create<spirv::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(paddingStride));
    auto operandExtent = builder.create<spirv::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(operandShape[i]));
    auto t1 =
        builder.create<spirv::IMulOp>(loc, i32Type, stride, operandExtent);
    auto bound = builder.create<spirv::IAddOp>(loc, i32Type, edgePadding, t1);

    // d_i
    auto dimIndex = affineExprCodegen.getValue(
        resultIndex.value(), builder.saveInsertionPoint(), loc);

    // d_i < (edge_padding_low[i] + stride * operand_shape[i])
    auto checkUb =
        builder.create<spirv::SLessThanOp>(loc, i1Type, dimIndex, bound);
    cond = builder.create<spirv::LogicalAndOp>(loc, i1Type, cond, checkUb);

    if (paddingLow != 0) {
      // d_i >= edge_padding_low[i]
      auto checkLb = builder.create<spirv::SGreaterThanEqualOp>(
          loc, i1Type, dimIndex, edgePadding);
      cond = builder.create<spirv::LogicalAndOp>(loc, i1Type, cond, checkLb);
    }

    if (paddingStride != 1) {
      // ((d_i - edge_padding_low[i]) % (interior_padding[i]+1) == 0)
      auto t1 = builder.create<spirv::ISubOp>(loc, dimIndex->getType(),
                                              dimIndex, edgePadding);
      auto t2 = builder.create<spirv::SModOp>(loc, t1.getResult()->getType(),
                                              t1, stride);
      auto checkStride = builder.create<spirv::IEqualOp>(loc, i1Type, t2, zero);
      cond =
          builder.create<spirv::LogicalAndOp>(loc, i1Type, cond, checkStride);
    }
  }
  Value *resultVal =
      builder.create<spirv::SelectOp>(loc, cond, operands[0], operands[1]);
  valueCache.setOperandDstValue(op->getResult(0), index, resultVal);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
