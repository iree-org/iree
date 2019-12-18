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

// IREE ops for working with buffers and buffer views.
// These are used by common transforms between the sequencer and interpreter and
// allow us to share some of the common lowering passes from other dialects.

#ifndef IREE_OPS
#define IREE_OPS

include "iree/compiler/IR/OpBase.td"

class IREE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class IREE_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREE_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

// TODO(b/134575149): determine if we want multiple constant op types.
def IREE_ConstantOp : IREE_PureOp<"constant", [
    AllShapesMatch<["value", "result"]>,
    AllElementTypesMatch<["value", "result"]>
]> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs IREEHL_MemRef:$result);

  // TODO(b/132296600): make tablegen follow the style guide.
  let extraClassDeclaration = [{
    Attribute getValue() { return value(); }
  }];

  let builders = [OpBuilder<"Builder*, OperationState&, ElementsAttr">];

  // TODO(b/134575149): enable folder when we store the correct type.
  // let hasFolder = 1;
}

// TODO(b/134671482): remove/move tensor_to_memref/memref_to_tensor.
def IREE_TensorToMemRefOp : IREE_PureOp<"tensor_to_memref", [
  SameOperandsAndResultShape, SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor);
  let results = (outs IREEHL_MemRef);

  let builders = [OpBuilder<"Builder*, OperationState&, Value*">];

  let hasFolder = 1;
}

// TODO(b/134671482): remove/move tensor_to_memref/memref_to_tensor.
def IREE_MemRefToTensorOp : IREE_PureOp<"memref_to_tensor", [
  SameOperandsAndResultShape, SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"Builder*, OperationState&, Value*">];

  let hasFolder = 1;
}

def IREE_ScalarToMemRefOp : IREE_PureOp<"scalar_to_memref", [
  SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_Element);
  let results = (outs IREEHL_AnyScalar);

  let builders = [OpBuilder<"Builder*, OperationState&, Value*">];

  let hasFolder = 1;
}

def IREE_MemRefToScalarOp : IREE_PureOp<"memref_to_scalar", [
  SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_AnyScalar);
  let results = (outs IREEHL_Element);

  let builders = [OpBuilder<"Builder*, OperationState&, Value*">];

  let hasFolder = 1;
}

def IREE_Workload : TensorOf<[AnyInteger]>;

def IREE_DispatchRegionOp : IREE_PureOp<"dispatch_region"> {
  let arguments = (ins
    IREE_Workload:$workload,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    Value *getWorkload() { return workload(); }
    Region& getBody() { return body(); }

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + 1; }
    unsigned getNumArgOperands() { return getNumOperands() - 1; }
    Value *getArgOperand(unsigned i) {
      return getOperand(mapArgOperandToOpOperand(i));
    }
    void setArgOperand(unsigned i, Value *arg) {
      setOperand(mapArgOperandToOpOperand(i), arg);
    }

    operand_iterator arg_operand_begin() {
      return operand_begin() + mapArgOperandToOpOperand(0);
    }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state,"
              "ArrayRef<Type> resultTypes, Value *workload,"
              "ValueRange args,"
              "ArrayRef<NamedAttribute> attributes = {}">,
  ];
}

def IREE_ReductionRegionOp : IREE_PureOp<"reduction_region", [
  SameVariadicOperandSize,
]> {
  let arguments = (ins
    IREE_Workload:$workload,
    Variadic<AnyType>:$operands,
    Variadic<AnyType>:$initial_values,
    OptionalAttr<I64ElementsAttr>:$dimensions,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<IREE_PaddingModeAttr>:$padding_mode
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    Value *getWorkload() { return workload(); }
    Region& getBody() { return body(); }

    bool isWindowed() {
      return window_dimensions().hasValue();
    }

    PaddingMode getPaddingMode() {
      return static_cast<PaddingMode>(padding_mode().getValue());
    }

    unsigned getNumReductionOperands() { return (getNumOperands() - 1) / 2; }
    operand_range getReductionOperands() { return getODSOperands(1); }
    operand_range getInitialValueOperands() { return getODSOperands(2); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state,"
    "ArrayRef<Type> resultTypes, Value *workload, ValueRange operands,"
    "ValueRange initialValues,"
    "ArrayRef<int64_t> dimensions,"
    "ArrayRef<NamedAttribute> attributes = {}">,
    OpBuilder<"Builder *builder, OperationState &state,"
    "ArrayRef<Type> resultTypes, Value *workload, ValueRange operands,"
    "ValueRange initialValues,"
    "ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,"
    "ArrayRef<int64_t> baseDilations, ArrayRef<int64_t> windowDilations,"
    "PaddingMode paddingMode,"
    "ArrayRef<NamedAttribute> attributes = {}">,
  ];
}

def IREE_ReturnOp : IREE_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
}

def IREE_LoadInputOp : IREE_PureOp<"load_input"> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs AnyType);
}

def IREE_StoreOutputOp : IREE_Op<"store_output"> {
  let arguments = (ins AnyType:$src, IREEHL_MemRef:$dst);
}

def IREE_StoreReduceOp : IREE_Op<"store_reduce"> {
  let arguments = (ins
    AnyType:$src,
    IREEHL_MemRef:$dst,
    FlatSymbolRefAttr:$reduction_fn
  );
}

#endif  // IREE_OPS
