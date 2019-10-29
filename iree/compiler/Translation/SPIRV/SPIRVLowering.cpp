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

//===- SPIRVLowering.cpp ---------------------------------------*- C++//-*-===//
//
// SPIR-V Code-generation for XLA-HLO Ops within IREE Dispatch functions
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/SPIRVLowering.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
LogicalResult ConstantOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index, ArrayRef<Value *>,
    AffineExprCodegen &affineExprCodegen, ValueCache &valueCache) const {
  auto constOp = cast<ConstantOp>(op);
  auto attr = constOp.value().dyn_cast<DenseElementsAttr>();
  if (!attr || !attr.isSplat()) {
    return op->emitError(
        "unhandled constant lowering unless value is a splat dense element "
        "attribute");
  }
  auto resultType = constOp.getResult()->getType();
  Type resultElemType;
  if (resultType.isIntOrFloat()) {
    resultElemType = resultType;
  } else if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    resultElemType = shapedType.getElementType();
  } else {
    return op->emitError("unhandled result type of constant : ") << resultType;
  }
  Attribute constVal = attr.getSplatValue();
  auto spirvConstOp =
      builder.create<spirv::ConstantOp>(op->getLoc(), resultElemType, constVal);
  valueCache.setOperandDstValue(constOp.getResult(), index,
                                spirvConstOp.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
LogicalResult CmpFOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache) const {
  if (operands.size() != 2) {
    return op->emitError("expected two operands in spir-v lowering of CmpFOp");
  }
  Operation *spirvOp = nullptr;
  auto opInfo = op->getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName());
  if (!opInfo) {
    return op->emitError("expected CmpFOp to contain ")
           << CmpFOp::getPredicateAttrName() << " attribute";
  }
  auto boolType = builder.getI1Type();
  auto predicateVal = static_cast<CmpFPredicate>(opInfo.getInt());
  switch (predicateVal) {
#define DISPATCH(caseLabel, opName)                                       \
  case caseLabel:                                                         \
    spirvOp = builder.create<opName>(op->getLoc(), boolType, operands[0], \
                                     operands[1]);                        \
    break;

    DISPATCH(CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    DISPATCH(CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

    default:
      return op->emitError("unhandled predicate attribute for SPIR-V lowering");
  }
  valueCache.setOperandDstValue(op->getResult(0), index, spirvOp->getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
LogicalResult ReturnOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineExprCodegen &affineExprCodegen,
    ValueCache &valueCache,
    DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
    ArrayRef<spirv::GlobalVariableOp> outputBuffers) const {
  auto returnOp = cast<ReturnOp>(op);
  if (returnOp.getNumOperands() != 1) {
    return returnOp.emitError(
        "unhandled lowering of return statement with multiple returns");
  }
  auto returnTensor = returnOp.getOperand(0);
  auto indices = affineExprCodegen.getIndices(returnTensor);
  if (indices.size() != 1) {
    return returnOp.emitError(
        "expected to compute a single element of the return tensor");
  }
  assert(outputBuffers.size() == 1 && "Expected a single output buffer");
  auto var = outputBuffers[0];
  auto ptr = genPointerOffset(builder, returnOp.getLoc(), affineExprCodegen,
                              indices[0], var);
  auto scalarVal = valueCache.getOperandDstValue(returnTensor, indices[0]);
  builder.create<spirv::StoreOp>(returnOp.getLoc(), ptr, scalarVal,
                                 /*memory_access = */ nullptr,
                                 /*alignment = */ nullptr);
  builder.create<spirv::ReturnOp>(returnOp.getLoc());
  return success();
}
}  // namespace iree_compiler
}  // namespace mlir
