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

#ifndef IREE_INTERPRETER_COMMON_OPS
#define IREE_INTERPRETER_COMMON_OPS

include "iree/compiler/Translation/Interpreter/IR/CommonBase.td"

class IREEInterp_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREEInterp_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class IREEInterp_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREEInterp_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

def IREEInterp_ConstantOp : IREEInterp_PureOp<"constant", [
    AllShapesMatch<["value", "result"]>,
    AllElementTypesMatch<["value", "result"]>
]> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs IREEHL_MemRef:$result);

  let extraClassDeclaration = [{
    Attribute getValue() { return value(); }
  }];

  let builders = [OpBuilder<"Builder*, OperationState&, ElementsAttr">];

  // TODO(b/134575149): enable folder when we store the correct type.
  // let hasFolder = 1;
}

def IREEInterp_TensorToMemRefOp : IREEInterp_PureOp<"tensor_to_memref", [
  SameOperandsAndResultShape, SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor);
  let results = (outs IREEHL_MemRef);

  let builders = [OpBuilder<"Builder*, OperationState&, Value ">];

  let hasFolder = 1;
}

def IREEInterp_MemRefToTensorOp : IREEInterp_PureOp<"memref_to_tensor", [
  SameOperandsAndResultShape, SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"Builder*, OperationState&, Value ">];

  let hasFolder = 1;
}

def IREEInterp_ScalarToMemRefOp : IREEInterp_PureOp<"scalar_to_memref", [
  SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_Element);
  let results = (outs IREEHL_AnyScalar);

  let builders = [OpBuilder<"Builder*, OperationState&, Value ">];

  let hasFolder = 1;
}

def IREEInterp_MemRefToScalarOp : IREEInterp_PureOp<"memref_to_scalar", [
  SameOperandsAndResultElementType
]> {
  let arguments = (ins IREEHL_AnyScalar);
  let results = (outs IREEHL_Element);

  let builders = [OpBuilder<"Builder*, OperationState&, Value ">];

  let hasFolder = 1;
}

#endif  // IREE_INTERPRETER_COMMON_OPS
