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

def IREE_NullOp : IREE_PureOp<"null"> {
  let summary = "a null type value";
  let description = [{
    Defines an SSA value that is lowered into dialects supporting
    null/undefined/optional/etc values.
  }];

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

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
// Executable ABI DO NOT ADD THINGS HERE
//===----------------------------------------------------------------------===//

// ***WARNING***: this is going away soon - please don't extend or add more
// things like this to this file. Ops should go to the dialect they relate to
// (HAL, in this case).
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
    OpBuilderDAG<(ins "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
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

  let builders = [
    OpBuilderDAG<(ins "Attribute":$value),
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

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//
// NOTE: this type is mostly just a placeholder. Ideally we'd make this
// immutable and have operations like iree.list.append/concat/etc the returned
// new SSA values. This would make optimizing the list usage much easier and
// enable hoisting/CSE of list access/mutation.

def IREE_ListSizeOp : IREE_Op<"list.size"> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    AnyList:$list
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREE_ListResizeOp : IREE_Op<"list.resize"> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain `new_size` elements. This will either truncate
    the list if the existing size is greater than `new_size` or extend the list
    with the default list value of the element type.
  }];

  let arguments = (ins
    AnyList:$list,
    Index:$new_size
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREE_ListGetOp : IREE_Op<"list.get"> {
  let summary = [{element accessor}];
  let description = [{
    Returns the value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    AnyList:$list,
    Index:$index
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "operands attr-dict `:` custom<ListType>(type($list), type($result))";

  let verifier = [{ return verify$cppClass(*this); }];
}

def IREE_ListSetOp : IREE_Op<"list.set"> {
  let summary = [{element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    AnyList:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "operands attr-dict `:` custom<ListType>(type($list), type($value))";

  let verifier = [{ return verify$cppClass(*this); }];
}

#endif  // IREE_DIALECT_IREE_OPS
