// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_OPS
#define IREE_DIALECT_UTIL_IR_UTIL_OPS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Util_Op<string mnemonic, list<Trait> traits = []> :
    Op<Util_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

class Util_PureOp<string mnemonic, list<Trait> traits = []> :
    Util_Op<mnemonic, !listconcat(traits, [Pure])>;

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
    reference types (like !util.buffer) that cannot be used with std.cmpi.
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
      "std::optional<std::pair<int64_t, int64_t>>":$integerRange
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

    std::optional<std::pair<int64_t, int64_t>> getIntegerRange();
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
    Variadic<Util_Range>:$operands
  );
  let results = (outs
    Util_Range:$result
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
    Variadic<Util_Range>:$operands
  );
  let results = (outs
    Util_Range:$result
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
    Variadic<Util_Range>:$offsets,
    Variadic<Util_Range>:$lengths
  );
  let results = (outs
    Util_Range:$min,
    Util_Range:$max
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

def Util_SizeOfOp : Util_PureOp<"sizeof"> {
  let summary = [{returns the size in bytes of a datatype}];
  let description = [{
    Most datatypes have a static size at all layers of the compilation stack.
    However, those that only have a size for certain lowering flows can be
    challenging. This op represents such sizes in a way that can be specialized
    later.

    Returns the size in bytes, rounded up to the next whole byte of the
    specified type. This op will fold to a constant index value for IntegerType
    and FloatType. All others are not folded.
  }];

  let arguments = (ins
    TypeAttr:$sizedType
  );
  let results = (outs
    Index:$size
  );

  let assemblyFormat = [{
    $sizedType attr-dict-with-keyword
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

def Util_OptimizationBarrierOp : Util_Op<"optimization_barrier", [
  SameOperandsAndResultType,
]> {
  let summary = "Prevents compiler optimizations across a value.";
  let description = [{
    Wraps any operands in an unoptimizable identity to prevent its results from
    being folded. It will be dropped during the final step in compilation and
    has no effect at runtime.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict
    ($operands^ `:` type($operands))?
  }];

  let builders = [
    OpBuilder<(ins
      "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let hasVerifier = 1;
}

def Util_UnfoldableConstantOp : Util_Op<"unfoldable_constant"> {
  let summary = "A constant that cannot be folded by the compiler.";
  let description = [{
    Similar to a std.constant, but is declared as having a side effect and has
    no folder. This is really just syntactic sugar as it is canonicalized to a
    std.constant wrapped in an util.optimization_barrier.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let hasCanonicalizer = 1;
}

def Util_UnreachableOp : Util_Op<"unreachable", [
    NoMemoryEffect,
    ReturnLike,
    Terminator
  ]> {
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
    Util_InitializerOpInterface,
  ]> {
  let summary = [{global initialization function}];
  let description = [{
    A function that is called in definition order upon module initialization.
    Must not load any globals that are defined or initialized after it in the
    module.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
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

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return {}; }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return {}; }

    LogicalResult verifyType() { return success(); }

    Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getCallableResults() { return {}; }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    /// Make symbol optional as this op has no symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let hasCanonicalizer = 1;
}

def Util_InitializerReturnOp : Util_Op<"initializer.return", [
  HasParent<"IREE::Util::InitializerOp">,
  Pure,
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
  Util_GlobalOpInterface,
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
    OptionalAttr<TypedAttrInterface>:$initial_value
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
      "std::optional<TypedAttr>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];

  let hasCanonicalizer = 1;
}

def Util_GlobalAddressOp : Util_PureOp<"global.address", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  SymbolUserOpInterface,
  Util_GlobalAddressOpInterface,
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
    $global attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalAddressOp(*this, symbolTable);
    }
  }];
}

def Util_GlobalLoadOp : Util_Op<"global.load", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  // HACK: works around the lack of symbol side effects in C++.
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SymbolUserOpInterface,
  Util_GlobalLoadOpInterface,
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
      "IREE::Util::GlobalOpInterface":$globalOp,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalLoadOp(*this, symbolTable);
    }
  }];
}

def Util_GlobalLoadIndirectOp : Util_Op<"global.load.indirect", [
  Util_GlobalLoadIndirectOpInterface,
]> {
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
    $global attr-dict `:` qualified(type($global)) `->` type($result)
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Util_GlobalStoreOp : Util_Op<"global.store", [
  SymbolUserOpInterface,
  Util_GlobalStoreOpInterface,
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

  let builders = [
    OpBuilder<(ins
      "Value":$value,
      "IREE::Util::GlobalOpInterface":$globalOp,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalStoreOp(*this, symbolTable);
    }
  }];

  let hasCanonicalizer = 1;
}

def Util_GlobalStoreIndirectOp : Util_Op<"global.store.indirect", [
  Util_GlobalStoreIndirectOpInterface,
]> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global variable.
  }];

  let arguments = (ins
    AnyType:$value,
    Arg<Util_AnyGlobalPtr, "", []>:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value) `->` qualified(type($global))
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// !util.list<T>
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
    Util_AnyListType:$result
  );

  let assemblyFormat = "($initial_capacity^)? attr-dict `:` qualified(type($result))";
}

def Util_ListSizeOp : Util_Op<"list.size", [MemoryEffects<[MemRead]>]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    Util_AnyListType:$list
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($list))";
}

def Util_ListResizeOp : Util_Op<"list.resize", [MemoryEffects<[MemWrite]>]> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain `new_size` elements. This will either truncate
    the list if the existing size is greater than `new_size` or extend the list
    with the default list value of the element type.
  }];

  let arguments = (ins
    Util_AnyListType:$list,
    Index:$new_size
  );

  let assemblyFormat = "operands attr-dict `:` qualified(type($list))";
}

def Util_ListGetOp : Util_Op<"list.get", [MemoryEffects<[MemRead]>]> {
  let summary = [{element accessor}];
  let description = [{
    Returns the value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    Util_AnyListType:$list,
    Index:$index
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` custom<ListTypeGet>(type($list), type($result))";

  let hasVerifier = 1;
}

def Util_ListSetOp : Util_Op<"list.set", [MemoryEffects<[MemWrite]>]> {
  let summary = [{element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    Util_AnyListType:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "$list `[` $index `]` `,` $value attr-dict `:` custom<ListTypeSet>(type($list), type($value))";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// !util.buffer
//===----------------------------------------------------------------------===//

def Util_BufferConstantOp : Util_PureOp<"buffer.constant", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{constant host-side byte buffer}];
  let description = [{
    Defines a compile-time byte buffer based on the given attribute value.
    The attribute will be serialized into the canonical IREE format for the
    chosen host target.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    Util_AnySerializableAttr:$value,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<StrAttr>:$mime_type
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    ($name^)? attr-dict `:` type($result) `=` $value
  }];

  let hasVerifier = 1;
}

def Util_BufferAllocOp : Util_PureOp<"buffer.alloc", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  MemoryEffects<[MemAlloc]>,
  Util_SizeAwareOp,
]> {
  let summary = [{allocates a buffer with undefined contents}];
  let description = [{
    Allocates a buffer with undefined contents. Consumers of the allocated
    result must assume nothing of the contents.
  }];

  let arguments = (ins
    Util_Size:$storage_size,
    OptionalAttr<IndexAttr>:$alignment
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    `uninitialized`
    attr-dict
    `:`
    type($result) `` `{` $storage_size `}`
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSize(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Util_BufferDeallocOp : Util_PureOp<"buffer.dealloc", [
  MemoryEffects<[MemFree]>,
  Util_SizeAwareOp,
]> {
  let summary = [{deallocates a buffer}];
  let description = [{
    Hints that the buffer contents can be discarded. Buffers are reference
    counted and other owners may keep it live beyond the dealloc.
  }];

  let arguments = (ins
    Util_BufferType:$operand,
    Util_Size:$operand_size
  );

  let assemblyFormat = [{
    $operand `:` type($operand) `{` $operand_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getOperandSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

def Util_BufferSliceOp : Util_PureOp<"buffer.slice", [
  AllTypesMatch<["source", "result"]>,
  MemoryEffects<[MemAlloc, MemRead]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{clones a subregion of a buffer}];
  let description = [{
    Returns a copy of the contents from the source buffer.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Util_Size:$source_size,
    Util_Offset:$source_offset,
    Util_Size:$result_size,
    OptionalAttr<IndexAttr>:$alignment
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` attr-dict `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];
}

def Util_BufferSubspanOp : Util_PureOp<"buffer.subspan", [
  AllTypesMatch<["source", "result"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<ViewLikeOpInterface>,
  Util_SizeAwareOp,
  Util_SubrangeOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{returns a reference to a subrange of a buffer}];
  let description = [{
    Returns a logical view into an underlying source buffer. This induces
    aliasing and multiple SSA values may allow access to the same underlying
    buffer storage.

    Subspans are a compiler-only concept and are propagated by an analysis pass
    to result in absolute offsets on accesses any place the subrange would have
    been used.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Util_Size:$source_size,
    Util_Offset:$source_offset,
    Util_Size:$result_size
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }

    Value getSubrangeResource() { return getSource(); }
    Value getSubrangeResourceSize() { return getSourceSize(); }
    Value getSubrangeOffset() { return getSourceOffset(); }
    Value getSubrangeLength() { return getResultSize(); }
    Value getSubrangeResult() { return getResult(); }

    // Walks up the use-def chain to find a subspan op that feeds into |value|.
    static IREE::Util::BufferSubspanOp findSubspanOp(Value value);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Util_BufferSizeOp : Util_PureOp<"buffer.size", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Util_SizeAwareOp,
]> {
  let summary = [{returns the total buffer storage size in bytes}];
  let description = [{
    Returns the total length of the buffer in bytes from its base offset.
  }];

  let arguments = (ins
    Util_BufferType:$operand
  );
  let results = (outs
    Util_Size:$result
  );

  let assemblyFormat = [{
    $operand
    `:` type($operand)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getResult(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Util_BufferStorageOp : Util_PureOp<"buffer.storage", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Util_SizeAwareOp,
]> {
  let summary = [{returns the underlying buffer storage range}];
  let description = [{
    Returns the buffer storage as a memref that must be offset and restricted to
    the returned range. The memref may be of any type and the user is
    responsible for ensuring that the reinterpret_cast-like behavior makes sense
    for the data they are accessing.
  }];

  let arguments = (ins
    Util_BufferType:$operand,
    Util_Size:$operand_size
  );
  let results = (outs
    AnyMemRef:$result,
    Util_Offset:$offset
  );

  let assemblyFormat = [{
    $operand
    `:` type($operand) `` `{` $operand_size `}` `->` `(` type($result) `,` type($offset) `)`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getOperandSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasCanonicalizer = 1;
}

def Util_BufferCopyOp : Util_Op<"buffer.copy", [
  MemoryEffects<[MemRead, MemWrite]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{copies a range of bytes between buffers}];
  let description = [{
    Copies a range of bytes as with memcpy (no overlapping).
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Util_Size:$source_size,
    Util_Offset:$source_offset,
    Util_BufferType:$target,
    Util_Size:$target_size,
    Util_Offset:$target_offset,
    Util_Size:$length
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `,`
    $target `[` $target_offset `]` `,`
    $length `:`
    type($source) `` `{` $source_size `}` `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return idx == 0 ? getSourceSize() : getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

def Util_BufferCompareOp : Util_PureOp<"buffer.compare", [
  MemoryEffects<[MemRead]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{compares a range of two buffers}];
  let description = [{
    Returns true if the two ranges are bitwise equivalent, somewhat like memcmp.
  }];

  let arguments = (ins
    Util_BufferType:$lhs,
    Util_Size:$lhs_size,
    Util_Offset:$lhs_offset,
    Util_BufferType:$rhs,
    Util_Size:$rhs_size,
    Util_Offset:$rhs_offset,
    Util_Size:$length
  );
  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    $lhs `[` $lhs_offset `]` `,`
    $rhs `[` $rhs_offset `]` `,`
    $length `:`
    type($lhs) `` `{` $lhs_size `}` `,`
    type($rhs) `` `{` $rhs_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return idx == 0 ? getLhsSize() : getRhsSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

def Util_BufferFillOp : Util_Op<"buffer.fill", [
  MemoryEffects<[MemWrite]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{fills a range of bytes with a value}];
  let description = [{
    Fills the contents of the buffer in the given byte range with a pattern.
    The offset and length must match the natural alignment of the pattern type.
  }];

  let arguments = (ins
    Util_FillPattern:$pattern,
    Util_BufferType:$target,
    Util_Size:$target_size,
    Util_Offset:$target_offset,
    Util_Size:$length
  );

  let assemblyFormat = [{
    $pattern `,`
    $target `[` $target_offset `for` $length `]` `:`
    type($pattern) `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

def Util_BufferLoadOp : Util_Op<"buffer.load", [
  MemoryEffects<[MemRead]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{loads a value from a buffer}];
  let description = [{
    Loads a value at a byte offset. Must be aligned to the natural size of the
    result type.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Util_Size:$source_size,
    Util_Offset:$source_offset,
    Util_Size:$length
  );
  let results = (outs
    Util_Primitive:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `for` $length `]`
    `:` type($source) `` `{` $source_size `}` `->` type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasFolder = 1;
}

def Util_BufferStoreOp : Util_Op<"buffer.store", [
  MemoryEffects<[MemWrite]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_SubrangeOperandOpInterface>,
]> {
  let summary = [{stores a value into a buffer}];
  let description = [{
    Stores a value at a byte offset. Must be aligned to the natural size of the
    source type.
  }];

  let arguments = (ins
    Util_Primitive:$source,
    Util_BufferType:$target,
    Util_Size:$target_size,
    Util_Offset:$target_offset,
    Util_Size:$length
  );

  let assemblyFormat = [{
    $source `,`
    $target `[` $target_offset `for` $length `]`
    `:` type($source) `->` type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];
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
