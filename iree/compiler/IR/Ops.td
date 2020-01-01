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

def IREE_ReturnOp : IREE_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
}

def IREE_LoadInputOp : IREE_PureOp<"load_input"> {
  let arguments = (ins IREE_MemRef:$src);
  let results = (outs AnyType);
}

def IREE_StoreOutputOp : IREE_Op<"store_output"> {
  let arguments = (ins AnyType:$src, IREE_MemRef:$dst);
}

def IREE_StoreReduceOp : IREE_Op<"store_reduce"> {
  let arguments = (ins
    AnyType:$src,
    IREE_MemRef:$dst,
    FlatSymbolRefAttr:$reduction_fn
  );
}

#endif  // IREE_OPS
