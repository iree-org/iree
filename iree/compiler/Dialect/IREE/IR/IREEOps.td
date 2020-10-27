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
include "mlir/Interfaces/SideEffectInterfaces.td"

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
// Byte buffers and host data
//===----------------------------------------------------------------------===//

def IREE_ByteBufferConstantOp : IREE_PureOp<"byte_buffer.constant"> {
  let summary = "constant host-side byte buffer";
  let description = [{
    Defines a compile-time byte buffer based on the given attribute value.
    The attribute will be serialized into the canonical IREE format for the
    chosen host target.
  }];

  let arguments = (ins
    ElementsAttr:$value
  );
  let results = (outs
    ByteBufferType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result) `=` $value
  }];
}

//===----------------------------------------------------------------------===//
// Executable ABI
//===----------------------------------------------------------------------===//

def IREE_PlaceholderOp : IREE_Op<"placeholder", [MemoryEffects<[MemAlloc]>]> {
  let summary = "A placeholder op to feed a value/buffer into computation";
  let description = [{
    This op is intended as a way to represent a value or buffer that will be fed
    into computation. It can host additional attributes related to the value or
    buffer. This op can be used for the cases where one need to feed in value or
    buffer to computation in a function but cannot do that via the function
    argument due to ABI or contract issues.
  }];

  let arguments = (ins StrAttr:$purpose);
  let results = (outs Res<AnyType, "", [MemAlloc]>:$output);

  let assemblyFormat = [{ `for` $purpose attr-dict `:` type($output) }];
}

def IREE_WorkgroupIdOp : IREE_PureOp<"workgroup_id"> {
  let summary = "Get grid index of an iree workgoup among a specific dimension.";
  let description = [{
    IREE workgroups are logically distributed among a hypergrid, each point sampled 
    from the grid corresponds to a logical thread. For example in a 3d grid case 
    the op quries (x, y, z) dimensions:

    ```mlir
     %0 = iree.workgroup_id {dimension = "x"} : index
     %1 = iree.workgroup_id {dimension = "y"} : index
     %2 = iree.workgroup_id {dimension = "z"} : index
    ```
  }];
  
  let arguments = (ins StrAttr:$dimension);
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def IREE_WorkgroupSizeOp: IREE_PureOp<"workgoup_size"> {
  let summary = "Get grid size of an iree thread among a specific dimension";
  let description = [{
    Get grid size of an iree workgroups among a specific dimension
  }];
  let arguments = (ins StrAttr:$dimension);
 let results = (outs Index:$result);

 let assemblyFormat = "attr-dict `:` type($result)";
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
    OpBuilder<"ValueRange operands, ArrayRef<NamedAttribute> attributes = {}">
  ];
}

// TODO(gcmn): It shouldn't be necessary to have both of these ops. Unify the
// approach here.

def IREE_DynamicShapeConstantOp : IREE_Op<"dynamic_shape_constant"> {
  let summary = "A tensor constant that can have dynamic dimensions";
  let description = [{
    Allows specifying a constant where the return value can erase shape
    information. This operation is declared as having side effects and has no
    folder, so will not be optimized away by the compiler. The underlying shape
    information should be hidden from the compiler and resolved at runtime.

    ```mlir
    %c = iree.dynamic_shape_constant tensor<2x2xf32> -> tensor<?x?xf32>
    %res = "mhlo.abs"(%c) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$value attr-dict `->` type($result)";
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
    "Attribute value",
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let hasCanonicalizer = 1;
}

def IREE_UnreachableOp : IREE_Op<"unreachable", [NoSideEffect, Terminator]> {
  let summary = [{unreachable assertion op}];
  let description = [{
    Signals to the compiler that the parent block should not be reachable.
    This may be converted into a runtime assertion, though ideally they are
    stripped during translation.

    ```mlir
    ^bb0:
      %true = constant true
      cond_br %true, ^bb2, ^bb1
    ^bb1:
      // Indicates that this branch should never be taken.
      iree.unreachable
    ^bb2:
      ...

    ```
  }];

  let assemblyFormat = "attr-dict";
}

#endif  // IREE_DIALECT_IREE_OPS
