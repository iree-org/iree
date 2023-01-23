//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_INTERFACES
#define TORCH_MLIR_DIALECT_TMTENSOR_INTERFACES

include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorBase.td"

// The interface is a subset of LinalgStructuredInterface.
def TMTensorInterface : OpInterface<"TMTensorOp"> {
  let methods = [
    //===------------------------------------------------------------------===//
    // Num input/output arguments handling.
    //===------------------------------------------------------------------===//
    // `inputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the input shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"getInputs",
      /*args=*/(ins)
    >,
    // These special methods rely on `inputs` and `outputs` being defined by
    // each op that wants to implement the LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputs().size();
      }]
    >,
    // `outputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the output shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"getOutputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs and outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputsAndOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumInputs() + getNumOutputs();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getInputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputs = getNumInputs();
        mlir::OpOperandVector result;
        result.reserve(numInputs);
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(numInputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getInputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getInputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of tensor type.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getInputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Output operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numOutputs = getNumOutputs();
        mlir::OpOperandVector result;
        result.reserve(numOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .drop_front(getNumInputs())
            .take_front(numOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getOutputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getOutputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getOutputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<MemRefType>",
      /*methodName=*/"getOutputBufferTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType>",
      /*methodName=*/"getOutputTensorTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"mlir::OpOperandVector",
      /*methodName=*/"getInputAndOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        mlir::OpOperandVector result;
        result.reserve(numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_front(numInputsAndOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an input tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an output tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOutputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the `opOperand` is a scalar value.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isScalar",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only MemRef input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasBufferSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(getInputOperands(), [&](OpOperand *opOperand) {
            return isScalar(opOperand) ||
              opOperand->get().getType().template isa<MemRefType>();
          }) &&
          llvm::all_of(getOutputOperands(), [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only RankedTensor input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasTensorSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return
          llvm::all_of(getInputOperands(), [&](OpOperand *opOperand) {
            return isScalar(opOperand) ||
              opOperand->get().getType().template isa<RankedTensorType>();
          }) &&
          llvm::all_of(getOutputOperands(), [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation. This
        does not change the balance between input, output_buffer and
        init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"clone",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        IRMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (Region &r : $_op->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
      }]
    >
  ];

  let extraClassDeclaration = [{
    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyTMTensorOpInterface($_op); }];
}

#endif  // TORCH_MLIR_DIALECT_TMTENSOR_INTERFACES
