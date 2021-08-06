// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_OPS
#define IREE_DIALECT_UTIL_IR_UTIL_OPS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Util_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Util_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class Util_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Util_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Byte buffers and host data
//===----------------------------------------------------------------------===//

def Util_NullOp : Util_PureOp<"null"> {
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

def Util_ByteBufferConstantOp : Util_PureOp<"byte_buffer.constant"> {
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
// Compiler hints
//===----------------------------------------------------------------------===//

def Util_DoNotOptimizeOp : Util_Op<"do_not_optimize"> {
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
    OpBuilder<(ins "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];
}

// TODO(gcmn): It shouldn't be necessary to have both of these ops. Unify the
// approach here.

def Util_DynamicShapeConstantOp : Util_Op<"dynamic_shape_constant"> {
  let summary = "A tensor constant that can have dynamic dimensions";
  let description = [{
    Allows specifying a constant where the return value can erase shape
    information. This operation is declared as having side effects and has no
    folder, so will not be optimized away by the compiler. The underlying shape
    information should be hidden from the compiler and resolved at runtime.

    ```mlir
    %c = util.dynamic_shape_constant tensor<2x2xf32> -> tensor<?x?xf32>
    %res = "mhlo.abs"(%c) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$value attr-dict `->` type($result)";
}

def Util_UnfoldableConstantOp : Util_Op<"unfoldable_constant"> {
  let summary = "A constant that cannot be folded by the compiler.";
  let description = [{
    Similar to a std.constant, but is declared as having a side effect and has
    no folder. This is really just syntactic sugar as it is canonicalized to a
    std.constant wrapped in an util.do_not_optimize.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let hasCanonicalizer = 1;
}

def Util_UnreachableOp : Util_Op<"unreachable", [NoSideEffect, Terminator]> {
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
      util.unreachable "shouldn't be here"
    ^bb2:
      ...

    ```
  }];

  let arguments = (ins StrAttr:$message);

  let assemblyFormat = "$message attr-dict";
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//
// NOTE: this type is mostly just a placeholder. Ideally we'd make this
// immutable and have operations like util.list.append/concat/etc the returned
// new SSA values. This would make optimizing the list usage much easier and
// enable hoisting/CSE of list access/mutation.

def Util_ListCreateOp : Util_PureOp<
    "list.create", [MemoryEffects<[MemAlloc]>]> {
  let summary = [{creates a new empty list}];
  let description = [{
    Creates a new empty list with an optional initial capacity.
  }];

  let arguments = (ins
    Optional<Index>:$initial_capacity
  );
  let results = (outs
    AnyList:$result
  );

  let assemblyFormat = "($initial_capacity^)? attr-dict `:` type($result)";
}

def Util_ListSizeOp : Util_Op<"list.size", [MemoryEffects<[MemRead]>]> {
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

def Util_ListResizeOp : Util_Op<"list.resize", [MemoryEffects<[MemWrite]>]> {
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

def Util_ListGetOp : Util_Op<"list.get", [MemoryEffects<[MemRead]>]> {
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

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` custom<ListTypeGet>(type($list), type($result))";

  let verifier = [{ return verify$cppClass(*this); }];
}

def Util_ListSetOp : Util_Op<"list.set", [MemoryEffects<[MemWrite]>]> {
  let summary = [{element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    AnyList:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "$list `[` $index `]` `,` $value attr-dict `:` custom<ListTypeSet>(type($list), type($value))";

  let verifier = [{ return verify$cppClass(*this); }];
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_OPS
