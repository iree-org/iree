// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_OPS
#define IREE_DIALECT_UTIL_IR_UTIL_OPS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Util_Op<string mnemonic, list<Trait> traits = []> :
    Op<Util_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class Util_PureOp<string mnemonic, list<Trait> traits = []> :
    Util_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Byte buffers and host data
//===----------------------------------------------------------------------===//

def Util_NullOp : Util_PureOp<"null"> {
  let summary = [{returns a null type value}];
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

def Util_CmpEQOp : Util_PureOp<"cmp.eq", [
  AllTypesMatch<["lhs", "rhs"]>,
  Commutative,
]> {
  let summary = [{compares two values for equality}];
  let description = [{
    Compares two operands for equality. This is intended for comparing IREE
    reference types (like !util.byte_buffer) that cannot be used with std.cmpi.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($lhs)
  }];

  let hasFolder = 1;
}

def Util_ByteBufferConstantOp : Util_PureOp<"byte_buffer.constant"> {
  let summary = "constant host-side byte buffer";
  let description = [{
    Defines a compile-time byte buffer based on the given attribute value.
    The attribute will be serialized into the canonical IREE format for the
    chosen host target.
  }];

  let arguments = (ins
    AnyAttr:$value,
    OptionalAttr<I64Attr>:$alignment
  );
  let results = (outs
    ByteBufferType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result) `=` $value
  }];
}

//===----------------------------------------------------------------------===//
// Data type conversions
//===----------------------------------------------------------------------===//

def Util_NumericOptionalNarrowOp : Util_PureOp<"numeric.optional_narrow", [
  SameOperandsAndResultType
]> {
  let summary = "memorializes an optional numeric narrowing that is valid";
  let description = [{
    Serves as a placeholder for points in the computation where an optional
    numeric narrowing can be performed without loss of information. Such ops
    can guide optimization passes wishing to perform precision reduction.

    In addition to the operand and result type, this op takes an additional
    `semantic_type` attribute representing the semantic target type which can
    be:
      * FloatType
      * Signed IntegerType
      * Unsigned IntegerType

    Note that this `semantic_type` must be a sign-carrying integer if using an
    integer type and cannot be IndexType (i.e. it can be used to indicate a
    possible narrowing of an IndexType to a specific integer).

    If the operand is a TensorType, then the result must be a TensorType. The
    `semantic_type` constrains the element type.

    Optionally, the minimum and maximum integer values (for integer semantic
    types) are tracked if known.
  }];

  let arguments = (ins
    AnyTypeOf<[Util_Element, Util_Tensor]>:$operand,
    TypeAttr:$semantic_type,
    OptionalAttr<APIntAttr>:$min_value,
    OptionalAttr<APIntAttr>:$max_value
  );
  let results = (outs
    AnyTypeOf<[Util_Element, Util_Tensor]>:$result
  );

  let assemblyFormat = [{
    $operand `:` type($operand) `as` $semantic_type attr-dict
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$operand,
      "Type":$type,
      "Optional<std::pair<int64_t, int64_t>>":$integerRange
    ),
    [{
      IntegerAttr minValueAttr;
      IntegerAttr maxValueAttr;
      if (integerRange) {
        minValueAttr = $_builder.getIntegerAttr(type, integerRange->first);
        maxValueAttr = $_builder.getIntegerAttr(type, integerRange->second);
      }
      build($_builder, $_state, operand.getType(), operand, TypeAttr::get(type),
        minValueAttr, maxValueAttr);
    }]>,
  ];

  let extraClassDeclaration = [{
    bool isSigned() {
      if (auto integerType = getType().dyn_cast<IntegerType>()) {
        return !integerType.isUnsigned();
      }
      return true;
    }

    Optional<std::pair<int64_t, int64_t>> getIntegerRange();
  }];
}

//===----------------------------------------------------------------------===//
// Range arithmetic
//===----------------------------------------------------------------------===//

def Util_RangeMinOp : Util_PureOp<"range.min", [
  SameOperandsAndResultType,
  SameVariadicOperandSize,
]> {
  let summary = [{returns the min of all values}];
  let description = [{
    Computes the min of a variadic list of operands. Though it's possible to
    express this with standard arithmetic this op enables more semantically
    meaningful folding/optimizations.
  }];

  let arguments = (ins
    Variadic<Util_Offset>:$operands
  );
  let results = (outs
    Util_Offset:$result
  );

  let assemblyFormat = [{
    $operands attr-dict `:` type($result)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Util_RangeMaxOp : Util_PureOp<"range.max", [
  SameOperandsAndResultType,
  SameVariadicOperandSize,
]> {
  let summary = [{returns the max of all values}];
  let description = [{
    Computes the max of a variadic list of operands. Though it's possible to
    express this with standard arithmetic this op enables more semantically
    meaningful folding/optimizations.
  }];

  let arguments = (ins
    Variadic<Util_Offset>:$operands
  );
  let results = (outs
    Util_Offset:$result
  );

  let assemblyFormat = [{
    $operands attr-dict `:` type($result)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Util_RangeExtentsOp : Util_PureOp<"range.extents", [
  SameOperandsAndResultType,
  SameVariadicOperandSize,
]> {
  let summary = [{returns the min/max of a union of a set of ranges}];
  let description = [{
    Computes min(offsets) and max(offsets + lengths). Though it's possible to
    express this with standard arithmetic this op enables more semantically
    meaningful folding/optimizations.
  }];

  let arguments = (ins
    Variadic<Util_Offset>:$offsets,
    Variadic<Util_Offset>:$lengths
  );
  let results = (outs
    Util_Offset:$min,
    Util_Offset:$max
  );

  let assemblyFormat = [{
    custom<RangeList>($offsets, $lengths) attr-dict `:` type($min)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Address/offset Arithmetic
//===----------------------------------------------------------------------===//

def Util_AlignOp : Util_PureOp<"align", [
  SameOperandsAndResultType
  ]> {
  let summary = "Aligns up to a power-of-two alignment if required";
  let description = [{
     Aligns |value| up to the given power-of-two |alignment| if required.
  }];

  let arguments = (ins
    SignlessIntegerLike:$value,
    SignlessIntegerLike:$alignment
  );

  let results = (outs
    SignlessIntegerLike:$result
  );

  let assemblyFormat = [{
    $value `,` $alignment attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$value,
      "int64_t":$alignment
    ),
    [{
      build($_builder, $_state, value.getType(), value,
       $_builder.createOrFold<arith::ConstantIndexOp>($_state.location, alignment));
    }]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

def Util_DoNotOptimizeOp : Util_Op<"do_not_optimize", [
  SameOperandsAndResultType,
]> {
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
    OpBuilder<(ins
      "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];
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
      %true = arith.constant true
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
// Structural ops
//===----------------------------------------------------------------------===//

def Util_InitializerOp : Util_Op<"initializer", [
    IsolatedFromAbove,
    FunctionOpInterface,
    CallableOpInterface,
  ]> {
  let summary = [{global initialization function}];
  let description = [{
    A function that is called in definition order upon module initialization.
    Must not load any globals that are defined or initialized after it in the
    module.
  }];

  let arguments = (ins
    TypeAttr:$type
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];

  let extraClassDeclaration = [{
    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock();
    Block *addBlock();

    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return {}; }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return {}; }

    LogicalResult verifyType() { return success(); }

    Region *getCallableRegion() { return &body(); }
    ArrayRef<Type> getCallableResults() { return {}; }
  }];

  let hasCanonicalizer = 1;
}

def Util_InitializerReturnOp : Util_Op<"initializer.return", [
  HasParent<"IREE::Util::InitializerOp">,
  NoSideEffect,
  ReturnLike,
  Terminator,
]> {
  let summary = [{return from a util.initializer}];
  let description = [{
    Returns control from an initializer function.
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

def Util_GlobalOp : Util_Op<"global", [
  Symbol,
]> {
  let summary = [{stateful global variable declaration}];
  let description = [{
    Declares a global variable that maintains its value across invocations.
    The value is tied to the execution context of the module and different
    contexts will have different variable storage.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<AnyAttr>:$initial_value
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name
    attr-dict
    custom<TypeOrAttr>($type, $initial_value)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      "Optional<Attribute>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];

  let extraClassDeclaration = [{
    StringRef getSymbolName() { return sym_name(); }

    bool isMutable() { return is_mutable(); }
    void makeMutable() { (*this)->setAttr("is_mutable", UnitAttr::get(getContext())); }

    Attribute getInitialValueAttr() { return initial_valueAttr(); }
    void clearInitialValue() { (*this)->removeAttr("initial_value"); }
    void setInitialValue(Attribute attr) { (*this)->setAttr("initial_value", attr); }
  }];

  let verifier = [{ return verifyGlobalOp(*this); }];

  let hasCanonicalizer = 1;
}

def Util_GlobalAddressOp : Util_PureOp<"global.address", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<Util_GlobalAccessorOpInterface>,
]> {
  let summary = [{returns an address reference to a global}];
  let description = [{
    Returns the address of a global as a typed reference. Can be used with the
    global load and store indirect ops.
  }];

  let arguments = (ins
    Util_GlobalRefAttr:$global
  );
  let results = (outs
    Util_AnyGlobalPtr:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    IREE::Util::GlobalOp getGlobalOp();
  }];
}

def Util_GlobalLoadOp : Util_Op<"global.load", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  // HACK: works around the lack of symbol side effects in C++.
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<Util_GlobalAccessorOpInterface>,
]> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a global variable value.
  }];

  let arguments = (ins
    Arg<Util_GlobalRefAttr, "", []>:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "IREE::Util::GlobalOp":$globalOp,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let extraClassDeclaration = [{
    IREE::Util::GlobalOp getGlobalOp();
    bool isGlobalImmutable();
  }];

  let verifier = [{ return verifyGlobalLoadOp(*this); }];
}

def Util_GlobalLoadIndirectOp : Util_Op<"global.load.indirect"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the global variable value.
  }];

  let arguments = (ins
    Arg<Util_AnyGlobalPtr, "", []>:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($global) `->` type($result)
  }];

  let verifier = [{ return verifyGlobalLoadIndirectOp(*this); }];

  let hasCanonicalizer = 1;
}

def Util_GlobalStoreOp : Util_Op<"global.store", [
  DeclareOpInterfaceMethods<Util_GlobalAccessorOpInterface>,
]> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global variable.
  }];

  let arguments = (ins
    AnyType:$value,
    Arg<Util_GlobalRefAttr, "", []>:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    IREE::Util::GlobalOp getGlobalOp();
  }];

  let verifier = [{ return verifyGlobalStoreOp(*this); }];

  let hasCanonicalizer = 1;
}

def Util_GlobalStoreIndirectOp : Util_Op<"global.store.indirect"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global variable.
  }];

  let arguments = (ins
    AnyType:$value,
    Arg<Util_AnyGlobalPtr, "", []>:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value) `->` type($global)
  }];

  let verifier = [{ return verifyGlobalStoreIndirectOp(*this); }];

  let hasCanonicalizer = 1;
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
    Util_AnyList:$result
  );

  let assemblyFormat = "($initial_capacity^)? attr-dict `:` type($result)";
}

def Util_ListSizeOp : Util_Op<"list.size", [MemoryEffects<[MemRead]>]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    Util_AnyList:$list
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
    Util_AnyList:$list,
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
    Util_AnyList:$list,
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
    Util_AnyList:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "$list `[` $index `]` `,` $value attr-dict `:` custom<ListTypeSet>(type($list), type($value))";

  let verifier = [{ return verify$cppClass(*this); }];
}

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

def Util_StatusCheckOkOp : Util_Op<"status.check_ok"> {
  let summary = [{raises a global failure if a status is not 'ok'}];
  let description = [{
    When the status is not 'ok' this signals a runtime failure that causes the
    entire active invocation - and possibly *all* in-flight and pending
    invocations - to fail with the given status. The status will be propagated
    back via the available runtime error handling mechanisms such as semaphores
    or synchronous invocation results.

    As the IREE execution model is deeply pipelined it's possible that failures
    have a latency between when they are emitted and when the application can
    observe the failure. It's also possible that other work that is in-flight
    or pending when the failure occurs will complete.
  }];

  let arguments = (ins
    Util_Status:$status,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $status (`,` $message^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$status, CArg<"StringRef", [{""}]>:$message),
    [{
      build(
        $_builder, $_state, status,
        message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
    }]>,
  ];
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_OPS
