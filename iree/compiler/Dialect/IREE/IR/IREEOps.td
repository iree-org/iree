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

#ifndef IREE_DIALECT_IREE_OPS
#define IREE_DIALECT_IREE_OPS

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "mlir/Interfaces/SideEffects.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class IREE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class IREE_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREE_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Executable ABI
//===----------------------------------------------------------------------===//

def IREE_LoadInputOp : IREE_PureOp<"load_input"> {
  let arguments = (ins IREE_MemRef:$src);
  let results = (outs AnyType);

  let assemblyFormat = [{
    `(` $src `:` type($src) `)` attr-dict `:` type(results)
  }];
}

def IREE_StoreOutputOp : IREE_Op<"store_output"> {
  let arguments = (ins AnyType:$src, IREE_MemRef:$dst);

  let assemblyFormat = [{
    `(` $src `:` type($src) `,` $dst `:` type($dst) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

def IREE_DoNotOptimizeOp : IREE_Op<"do_not_optimize"> {
  let summary = "Prevents compiler optimizations of a value.";
  let description = [{
    Wraps any operands in an unoptimizable identity. This operation is declared
    as having side effects, so no compiler optimizations will be able to reason
    about it. This prevents its results from being folded. It will be dropped as
    the final step in compilation.
  }];
  let arguments = (ins Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>:$results);
  let verifier = [{ return verify$cppClass(*this); }];
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}">
  ];
}

def IREE_UnfoldableConstantOp : IREE_Op<"unfoldable_constant"> {
  let summary = "A constant that cannot be folded by the compiler.";
  let description = [{
    Similar to a std.constant, but is declared as having a side effect and has
    no folder. This is really just syntactic sugar as it is canonicalized to a
    std.constant wrapped in an iree.do_not_optimize.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_IREE_OPS
