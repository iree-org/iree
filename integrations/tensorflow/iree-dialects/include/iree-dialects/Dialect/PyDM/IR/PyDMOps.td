// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_OPS_TD
#define IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_OPS_TD

include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Non-variable assignment
//===----------------------------------------------------------------------===//

def IREEPyDM_AssignSubscriptOp : IREEPyDM_Op<"assign_subscript", [
    MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Assign to a subscript";
  let description = [{
    This implements the subscripted assignment form. In Python this is:

    ```
      foo[index] = bar
    ```
  }];
  let arguments = (ins
    IREEPyDM_AnyValueType:$lhs,
    IREEPyDM_AnyValueType:$slice,
    IREEPyDM_AnyValueType:$rhs
  );
  let results = (outs IREEPyDM_ExceptionResultType:$exc_result);
  let assemblyFormat = [{
    $lhs `[` $slice `]` `=` $rhs `:` type(operands) attr-dict
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Variable access
//===----------------------------------------------------------------------===//

def IREEPyDM_AllocFreeVarOp : IREEPyDM_PureOp<"alloc_free_var", [
    MemoryEffects<[MemAlloc]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Declares a 'free variable' by name and instance number";
  let description = [{
    Free variables represent allocated storage for a single value. By default
    it is a variant which can store anything.
  }];

  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<IndexAttr>:$index);
  let results = (outs IREEPyDM_FreeVarRef:$free_var);
  let assemblyFormat = [{
    $name (`[` $index^ `]`)? `->` type($free_var) attr-dict
  }];
}

def IREEPyDM_LoadVarOp : IREEPyDM_PureOp<"load_var", [MemoryEffects<[MemRead]>]> {
  let summary = "Loads a value from a variable";
  let description = [{
    Loads a value from a variables. The value is implicitly cast to the given
    result type, which must be legal.
  }];

  let arguments = (ins IREEPyDM_AnyVarRef:$var);
  let results = (outs IREEPyDM_AnyValueType);

  let assemblyFormat = [{
    $var `:` type($var) `->` type(results) attr-dict
  }];
}

def IREEPyDM_StoreVarOp : IREEPyDM_Op<"store_var", [MemoryEffects<[MemWrite]>]> {
  let summary = "Stores a value to a variable";

  let arguments = (ins
    IREEPyDM_AnyVarRef:$var,
    IREEPyDM_AnyValueType:$value);
  let assemblyFormat = [{
    $var `=` $value `:` type(operands) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// TODO: Move arg attributes to arguments, and generally rework the free_vars,
// cell_vars attributes (with better modeling, they may not be needed at all).
def IREEPyDM_FuncOp : IREEPyDM_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    CallableOpInterface,
    Symbol,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>]> {
  let summary = "Python func";
  let description = [{
    Python functions map arguments to results and have the following additional
    characteristics:
      - Have arguments that are either !object or primitive types. Typical user
        funcs will just be !object based unless if refined.
      - Returns a (LogicalResult, value). The LogicalResult will be expanded in
        the future to be a full exception record, and the value is an
        object/primitive (which may be a tuple/sequence if unpacking).
      - Have an explicit list of free and cell variable names which can be
        accessed from within the function.
      - ... other things as needed.
      - Entry block with arguments matching the function arguments.

    These functions are CFG based, functioning as a Block with a single implicit
    exception handler which matches all exceptions and exits with a failing
    status on failure.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<StrArrayAttr>:$arg_names,
                       OptionalAttr<StrArrayAttr>:$free_vars,
                       OptionalAttr<StrArrayAttr>:$cell_vars,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock();

    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    /// Returns the type of this function.
    FunctionType getType() {
      return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Returns the python return type of the function (second return type).
    Type getPyReturnType() {
      return getType().getResult(1);
    }

    /// Hook for Trait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return &body(); }
    ArrayRef<Type> getCallableResults() {
      return getType().getResults();
    }

    /// Defines SymbolOpInterface::isDeclaration().
    bool isDeclaration() {
      return body().empty();
    }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "FunctionType":$type), [{
      build($_builder, $_state, name, TypeAttr::get(type),
            nullptr, nullptr, nullptr, nullptr);
    }]>
  ];

  // TODO: Enforce invariants.
  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

def IREEPyDM_ReturnOp : IREEPyDM_Op<"return", [
    NoSideEffect,
    HasParent<"FuncOp">,
    ReturnLike,
    Terminator]> {
  let summary = "Successful return from a Python function";
  let description = [{
    Returns a value from a Python function.
  }];

  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def IREEPyDM_RaiseOnFailureOp : IREEPyDM_Op<"raise_on_failure", [
    HasParent<"FuncOp">]> {
  let summary = "Performs a non-local exit on failure of an ExceptionResult";
  let description = [{
    This op handles the vast majority of cases where a failure simply needs
    to be propagated to the next most frame (typically returning it from
    a function).

    If the `exc_result` represents a failure, control will not proceed
    past this operation.
  }];

  let arguments = (ins IREEPyDM_ExceptionResultType:$exc_result);
  let assemblyFormat = [{
    $exc_result `:` type($exc_result) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def IREEPyDM_CallOp : IREEPyDM_Op<"call", [
    CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call a `func` op";
  let description = [{
    This is the most primitive call operation that all other static calls
    decay into.

  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<IREEPyDM_AnyValueType>:$operands);
  let results = (outs
      IREEPyDM_ExceptionResultType:$exc_result,
      IREEPyDM_AnyValueType:$result);

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def IREEPyDM_PatternMatchCallOp : IREEPyDM_Op<"pattern_match_call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Combines multiple functions together for generic dispatch";
  let description = [{
    This op enables specialized arity and type based dispatch via simple
    pattern matching. It is generally used by the implementation to provide
    both fully generic, runtime based implementations with the option to
    select a specialization.

    Alternatives are split into two groups:
      - 'generic': Provides a maximally generic fallback implementation.
        Generally an alternative will be matched purely based on arity and
        structure of arguments. First match wins.
      - 'specific': Provides a specific, strongly typed implementation. Matching
        is by full type signature. First match wins.

    Generally, during compilation, such calls will decay into regular calls
    to one of the referenced, backing functions. How this is
    done depends on phase:
      - During canonicalization: A match to a specific function will be taken
        immediately.
      - After type inference: Any leftovers must be matched to a generic (by
        an explicit pass) for runtime resolution.
  }];

  let arguments = (ins IREEPyDM_FlatSymbolRefArrayAttr:$generic_match,
                       IREEPyDM_FlatSymbolRefArrayAttr:$specific_match,
                       Variadic<IREEPyDM_AnyValueType>:$operands);
  let results = (outs
      IREEPyDM_ExceptionResultType:$exc_result,
      IREEPyDM_AnyValueType:$result);

  let assemblyFormat = [{
    `(` $operands `)` `:` functional-type($operands, results)
    `matching` `generic` $generic_match `specific` $specific_match
    attr-dict
  }];
}

def IREEPyDM_DynamicCallOp : IREEPyDM_PureOp<"dynamic_call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Dynamic dispatch to a `func`.";
  let description = [{
    Performs full dynamic dispatch to a function. Most imports start in this
    form and are progressively refined to something more specific as more
    information is known. In general, for well-formed programs, there should
    not be any `dynamic_call`s left at the lowest levels of the compiler.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<IREEPyDM_AnyValueType>:$operands);
  let results = (outs
      IREEPyDM_ExceptionResultType:$exc_result,
      IREEPyDM_ObjectType:$result);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Boxing/unboxing
//===----------------------------------------------------------------------===//

def IREEPyDM_BoxOp : IREEPyDM_PureOp<"box", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Boxes a primitive into an object";
  let description = [{
    Given a PrimitiveType, boxes it into an appropriate !object, establishing
    identity. For value typed primitives, it is always safe to box, but for
    reference primitives, the providence must be tracked and the original boxed
    value used (vs boxing a new one). Failure to do so will result in aliased
    objects.

    Note that this operation allows to box object->object but canonicalizes
    away in such a case (this is a convenience for IR construction).
  }];

  let arguments = (ins
    IREEPyDM_AnyValueType:$primitive
  );
  let results = (outs
    IREEPyDM_ObjectType:$object
  );

  let assemblyFormat = [{
    $primitive `:` type($primitive)  `->` type($object) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def IREEPyDM_UnboxOp : IREEPyDM_PureOp<"unbox", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Unboxes an object to a specific primitive type";
  let description = [{
    Unboxes to a primitive, returning a failure result and a default initialized
    primitive value on failure to unbox.
  }];

  let arguments = (ins
    IREEPyDM_ObjectType:$object
  );
  let results = (outs
    IREEPyDM_ExceptionResultType:$exc_result,
    IREEPyDM_AnyPrimitiveType:$primitive
  );

  let assemblyFormat = [{
    $object `:` type($object) `->` type($primitive) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Value constructors
//===----------------------------------------------------------------------===//

def IREEPyDM_ConstantOp : IREEPyDM_PureOp<"constant", [ConstantLike]> {
  let summary = "Constants for value types";
  let description = [{
    This op supports immutable value types that have direct coding as MLIR
    attributes:
      IntType -> IntegerAttr<i64>
      RealType -> FloatAttr<double>
      StrType -> StringAttr
      BytesType -> StringAttr
      BoolType -> IntegerAttr<i1>
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let assemblyFormat = [{
    $value `->` type(results) attr-dict
  }];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];
  let hasFolder = 1;
}

def IREEPyDM_NoneOp : IREEPyDM_PureOp<"none", [ConstantLike]> {
  let summary = "Gets the singleton NoneType primitive value";
  let results = (outs IREEPyDM_NoneType:$value);
  let assemblyFormat = [{
    attr-dict
  }];
  let hasFolder = 1;
}

// TODO: Make ConstantLike
def IREEPyDM_FailureOp : IREEPyDM_PureOp<"failure", []> {
  let summary = "Generates a constant failure ExceptionResult";
  let results = (outs IREEPyDM_ExceptionResultType);
  let assemblyFormat = [{
    `->` type(results) attr-dict
  }];
}

def IREEPyDM_SuccessOp : IREEPyDM_PureOp<"success", [ConstantLike]> {
  let summary = "Generates a constant success ExceptionResult";
  let description = [{
    A successful ExceptionResultType folds to a UnitAttr.
  }];
  let results = (outs IREEPyDM_ExceptionResultType);
  let assemblyFormat = [{
    `->` type(results) attr-dict
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

def IREEPyDM_AsBoolOp : IREEPyDM_PureOp<"as_bool"> {
  let summary = "Evaluates an arbitrary value for its truthiness";
  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let results = (outs IREEPyDM_BoolType);
  let assemblyFormat = [{
    $value `:` type($value) `->` type(results) attr-dict
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def IREEPyDM_BoolToPredOp : IREEPyDM_PureOp<"bool_to_pred"> {
  let summary = "Extracts a pred (i1) value from a BoolType";
  let description = [{
    This dialect does not use native MLIR IntegerTypes. This is used for
    bridging to other dialects.
  }];
  let arguments = (ins IREEPyDM_BoolType:$value);
  let results = (outs I1);
  let assemblyFormat = [{
    $value attr-dict
  }];
  let hasFolder = 1;
}

def IREEPyDM_MakeListOp : IREEPyDM_PureOp<"make_list", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Makes a list from a static list of values";
  let description = [{
    Constructs a list from elements.
  }];
  let arguments = (ins Variadic<IREEPyDM_AnyValueType>:$elements);
  let results = (outs IREEPyDM_ListType:$list);
  let assemblyFormat = [{
    ($elements^ `:` type($elements))? `->` type(results) attr-dict
  }];
  let hasVerifier = 1;
}

def IREEPyDM_MakeTupleOp : IREEPyDM_PureOp<"make_tuple"> {
  let summary = "Makes a tuple from a static list of values";
  let description = [{
    Used for static construction of a tuple when the exact nature is known.
  }];
  let arguments = (ins Variadic<IREEPyDM_AnyValueType>:$slots);
  let results = (outs IREEPyDM_TupleType:$tuple);
  let assemblyFormat = [{
    ($slots^ `:` type($slots))? `->` type(results) attr-dict
  }];
}

def IREEPyDM_DynamicUnpackOp : IREEPyDM_PureOp<"dynamic_unpack"> {
  let summary = "Dynamically unpacks a tuple/sequence into a fixed number of values";
  let description = [{
    This is the most generic form of sequence unpacking for use in extracting
    values, when the arity is statically known. When more specific types are
    known, this op will generally decay into a more statically verifiable
    form or be elided entirely.

    This operation allows a tuple to be unpacked to slots of arbitrary type,
    but no type checking is done. Typically, slot types will be of `ObjectType`
    at the outset of program capture, and they may only be narrowed by
    optimizations which have adequately proven correctness.
  }];
  let arguments = (ins IREEPyDM_AnyValueType:$sequence);
  let results = (outs
    IREEPyDM_ExceptionResultType:$exc_result,
    Variadic<IREEPyDM_AnyValueType>:$slots);
  let assemblyFormat = [{
    $sequence `:` type($sequence) `->` type($exc_result) `,` `[` type($slots) `]` attr-dict
  }];
}

def IREEPyDM_GetTypeCodeOp : IREEPyDM_PureOp<"get_type_code"> {
  let summary = "Gets the type code (BuiltinTypeCode) of an arbitrary value";
  let description = [{
    Every PyDM value has a type that can be represented by a type code.
    See the `BuiltinTypeCode` enumeration for constants and encodings.
  }];
  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let results = (outs IREEPyDM_IntegerType);
  let assemblyFormat = [{
    $value `:` type($value) `->` type(results) attr-dict
  }];
}

def IREEPyDM_GetNumericPromotionOrderOp : IREEPyDM_PureOp<"get_numeric_promotion_order"> {
  let summary = "Gets the promotion order of a value";
  let description = [{
    Every numeric type is assigned a promotion order which:
      - When equal, means that no promotion is needed.
      - Is greater then another, means the both should be promoted based on
        the greater.
    TODO: There are additional special considerations for unsigned->signed
    promotion. This order only represents signed.

    Non-numeric values have a promotion order of 0.
  }];
  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let results = (outs IREEPyDM_IntegerType);
  let assemblyFormat = [{
    $value `:` type($value) `->` type(results) attr-dict
  }];
}

def IREEPyDM_StaticInfoCastOp : IREEPyDM_PureOp<"static_info_cast"> {
  let summary = "Adds or removes static type information";
  let description = [{
    This is often inserted as an artifact of type refinement where static
    information is added to a type locally and a follow-on step is required
    to resolve it.
  }];
  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let results = (outs IREEPyDM_AnyValueType);
  let assemblyFormat = [{
    $value `:` type($value) `->` type(results) attr-dict
  }];
  // TODO: Add verifier.
  // let verifier = 1;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def IREEPyDM_SelectOp : IREEPyDM_PureOp<"select", [
  AllTypesMatch<["true_value", "false_value", "result"]>
    ]> {
  let summary = "Select a true or false value based on condition";
  let arguments = (ins
      IREEPyDM_BoolType:$condition,
      IREEPyDM_AnyValueType:$true_value,
      IREEPyDM_AnyValueType:$false_value);
  let results = (outs IREEPyDM_AnyValueType:$result);
  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value `:` type($result) attr-dict
  }];
  let hasFolder = 1;
}

def IREEPyDM_ExprStatementDiscardOp : IREEPyDM_Op<"expr_statement_discard"> {
  let summary = "Anchors an expression evaluated as a statement";
  let description = [{
    This op is used early during import to keep a statement-evaluated expression
    live until more effect information is available to anchor it properly.
  }];
  let arguments = (ins IREEPyDM_AnyValueType:$value);
  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def FunctionalIfOp : IREEPyDM_Op<"functional_if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects,
    NoRegionArguments,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>]> {
  let summary = "A functional if construct";
  let description = [{
    This is similar to `scf.if` but adapted for this dialect. It has a required
    then region and an optional else region. Types yielded from both must
    match the result types of the `functional_if` op.
  }];
  let arguments = (ins IREEPyDM_BoolType:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def YieldOp : IREEPyDM_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                    ParentOneOf<["FunctionalIfOp"]>]> {
  let summary = "Yields a value from a functional control flow region";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

//===----------------------------------------------------------------------===//
// Computation
//===----------------------------------------------------------------------===//

def IREEPyDM_DynamicBinaryPromoteOp : IREEPyDM_PureOp<"dynamic_binary_promote", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Promotes two arguments that may be of numeric types";
  let description = [{
    Takes two values of arbitrary type which may be input to a following
    binary arithmetic operation. If they are both numeric, returns the two
    arguments, promoted to the bounding, common numeric type. Otherwise,
    returns the inputs as-is.

    This op will always produce boxed, type erased values since that is always
    legal for runtime evaluation. However, for known input types, it will
    canonicalize to various identities or fixed numerics, making the trailing
    boxing trivial to type propagate past.

    Note that this operation could be implemented with language-level pattern
    matching, but it is both very frequently used and complicated from a
    type system perspective. As such, we make it a black box with custom
    canonicalization and lowering.
  }];

  let arguments = (ins
    IREEPyDM_AnyValueType:$left,
    IREEPyDM_AnyValueType:$right);
  let results = (outs
    IREEPyDM_GenericObjectType:$left_prime,
    IREEPyDM_GenericObjectType:$right_prime);
  let assemblyFormat = [{
    $left `,` $right `:` type($left) `,` type($right) attr-dict
  }];
  let hasCanonicalizer = 1;
}

def IREEPyDM_PromoteNumericOp : IREEPyDM_PureOp<"promote_numeric"> {
  let summary = "Promotes one numeric type to another higher on the hierarchy";
  let description = [{
    Given a numeric value of lower promotion order, promotes it to a higher
    order.
  }];
  let arguments = (ins IREEPyDM_AnyNumericType:$input);
  let results = (outs IREEPyDM_AnyNumericType);
  let assemblyFormat = [{
    $input `:` type(operands) `->` type(results) attr-dict
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def IREEPyDM_ApplyBinaryOp : IREEPyDM_PureOp<"apply_binary", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface, ["refineResultTypes"]>
  ]> {
  let summary = "Applies one of Python's binary operations";
  let description = [{
    The operation to apply is specified as per the data model:
      https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    Here it is presented with leading and trailing double underscores (i.e.
    "add", "sub", etc).

    Numeric types must be promoted to a common type prior to application.
  }];
  let arguments = (ins
    StrAttr:$dunder_name,
    IREEPyDM_AnyValueType:$left,
    IREEPyDM_AnyValueType:$right);
  let results = (outs IREEPyDM_AnyValueType:$result);
  let assemblyFormat = [{
    $dunder_name `,` $left `,` $right `:` type(operands) `->` type(results) attr-dict
  }];
  let hasCanonicalizer = 1;
}

def IREEPyDM_ApplyCompareOp : IREEPyDM_PureOp<"apply_compare", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Performs a binary comparison";
  let description = [{
    Performs a comparison between two operands.

    Op name is based on the dunder name of the rich comparison ops in the
    data model:
      "lt", "le", "eq", "ne", "gt", "ge"
    With extensions for those that do not have a dunder name:
      "is", "isnot", "in", "notin"

    Numeric types must be promoted to a common type prior to application.
  }];
  let arguments = (ins
    StrAttr:$dunder_name,
    IREEPyDM_AnyValueType:$left,
    IREEPyDM_AnyValueType:$right);
  let results = (outs IREEPyDM_BoolType:$result);
  let assemblyFormat = [{
    $dunder_name `,` $left `,` $right `:` type(operands) attr-dict
  }];
  let hasCanonicalizer = 1;
}

def IREEPyDM_NegOp : IREEPyDM_PureOp<"neg", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface, ["refineResultTypes"]>
  ]> {
  let summary = "Unary negation";
  let description = [{
    Performs unary negation (i.e. `-3`) in a type independent way.
  }];
  let arguments = (ins
    IREEPyDM_AnyValueType:$value);
  let results = (outs IREEPyDM_AnyValueType:$result);
  let assemblyFormat = [{
    $value `:` type($value) `->` type($result) attr-dict
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Object method/feature access
//===----------------------------------------------------------------------===//

def IREEPyDM_SubscriptOp : IREEPyDM_PureOp<"subscript", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface>
  ]> {
  let summary = "Evaluates a subscript operation against a value";
  let description = [{
    This fully dynamic form of subscript can handle an arbitrarily general
    subscript operation, consisting of either a single numeric index or a Slice.
    In typical usage, it should canonicalize into specific special form.
  }];
  let arguments = (ins
    IREEPyDM_AnyValueType:$value,
    IREEPyDM_AnyValueType:$slice);
  let results = (outs
    IREEPyDM_ExceptionResultType:$exc_result,
    IREEPyDM_AnyValueType:$result);
  let assemblyFormat = [{
    $value `[` $slice `]` `:` type(operands) `->` type($result) attr-dict
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Sequence ops
//===----------------------------------------------------------------------===//

def IREEPyDM_SequenceCloneOp : IREEPyDM_PureOp<"sequence_clone", [
    DeclareOpInterfaceMethods<TypeRefinableOpInterface, ["refineResultTypes"]>
  ]> {
  let summary = "Clones a sequence, potentially duplicating it";
  let description = [{
    This is intended to be used as a lowering for higher level ops and as a
    primitive to be used by the runtime library. Given a sequence and a `count`,
    it produces a sequence of the same fundamental type and storage,
    duplicated `count` times.

    It is expected that the runtime can emit direct lowerings for variations
    operating on builtin sequence types.

    The input and result sequence type can be different (i.e. in case if the
    type bounds for elements is different).
  }];
  let arguments = (ins
    IREEPyDM_AnyValueType:$sequence,
    IREEPyDM_IntegerType:$count
  );
  let results = (outs
    IREEPyDM_AnyValueType
  );
  let assemblyFormat = [{
    $sequence `*` $count `:` type(operands) `->` type(results) attr-dict
  }];
}

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_OPS_TD
