// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_OPS_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_OPS_TD

include "iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// TODO: Move arg attributes to arguments, and generally rework the free_vars,
// cell_vars attributes (with better modeling, they may not be needed at all).
def IREEPyDM_FuncOp : IREEPyDM_Op<"func", [
    IsolatedFromAbove,
    FunctionLike,
    CallableOpInterface,
    Symbol]> {
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
                       StrArrayAttr:$arg_names,
                       StrArrayAttr:$free_vars,
                       StrArrayAttr:$cell_vars,
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

    /// Hook for OpTrait::FunctionLike, returns the number of function
    /// arguments. Depends on the type attribute being correct as checked by
    /// verifyType.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Hook for OpTrait::FunctionLike, returns the number of function results.
    /// Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return &body(); }
    ArrayRef<Type> getCallableResults() {
      return getType().getResults();
    }
  }];

  let parser = [{ return ::parseFuncOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
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

def IREEPyDM_BoxOp : IREEPyDM_PureOp<"box"> {
  let summary = "Boxes a primitive into an object";
  let description = [{
    Given a PrimitiveType, boxes it into an appropriate !object, establishing
    identity. For value typed primitives, it is always safe to box, but for
    reference primitives, the providence must be tracked and the original boxed
    value used (vs boxing a new one). Failure to do so will result in aliased
    objects.
  }];

  let arguments = (ins
    IREEPyDM_AnyPrimitiveType:$primitive
  );
  let results = (outs
    IREEPyDM_ObjectType:$object
  );

  let assemblyFormat = [{
    $primitive `:` type($primitive)  `->` type($object) attr-dict
  }];
}

def IREEPyDM_UnboxOp : IREEPyDM_PureOp<"unbox"> {
  let summary = "Unboxes an object to a specific primitive type";
  let description = [{
    Unboxes to a primitive, returning a failure result and a default initialized
    primitive value on failure to unbox.
  }];

  let arguments = (ins
    IREEPyDM_ObjectType:$object
  );
  let results = (outs
    IREEPyDM_ExceptionResultType:$status,
    IREEPyDM_AnyPrimitiveType:$primitive
  );

  let assemblyFormat = [{
    $object `:` type($object) `->` type($primitive) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Name access
// While the CPython nomenclature is a bit inconsistent, we follow it where
// it makes sense. Names are split into a few categories:
//   - Free Variables: Local variables that are not visible outside of the
//     containing (actual, not lexical) function. CPython bytecode refers
//     to operations on these as "FAST" (i.e. LOAD_FAST, STORE_FAST, DEL_FAST).
//   - Cell Variables: Variables that exist as part of the function closure
//     and are resolved through a level of indirection (cell).
//   - Global Variables: Variables that are resolved via the function's
//     globals().
// At this level, all operations are done by name (at the CPython instruction
// level, it is by ordinal).
//===----------------------------------------------------------------------===//

// TODO: Consider adding a dedicated free_var op to bind free variables and
// move the string attribute there.
def IREEPyDM_LoadFreeVarOp : IREEPyDM_PureOp<"load_free_var"> {
  let summary = "Loads a boxed free variable";
  let description = [{
    Loads the boxed object for a free variable. This is not failable at
    runtime, and if the slot is not initialized, it will contain a special
    NotInitialized primitive which cannot convert to anything else.

    When importing Python programs, they must use boxed accessors to free
    variable storage. Further analysis and transformation can promote these
    to unboxed access.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs IREEPyDM_ObjectType:$value);

  let assemblyFormat = [{
    $name `->` type($value) attr-dict
  }];
}

def IREEPyDM_StoreFreeVarOp : IREEPyDM_Op<"store_free_var"> {
  let summary = "Stores a boxed free variable";
  let description = [{
    Stores a boxed value to a free variable slot.
  }];

  let arguments = (ins StrAttr:$name, IREEPyDM_ObjectType:$value);

  let assemblyFormat = [{
    $name `,` $value `:` type($value) attr-dict
  }];
}

def IREEPyDM_LoadFreeVarUnboxedOp : IREEPyDM_PureOp<"load_free_var_unboxed"> {
  let summary = "Loads an unboxed free variable";
  let description = [{
    Loads an unboxed value from a free variable slot. This will not be
    emitted in regular programs but can be used in intrinsics and by
    optimizations to remove boxing when types are guaranteed. It is a
    program error to load a mismatched type and will produce undefined
    behavior.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs IREEPyDM_PrimitiveType:$value);

  let assemblyFormat = [{
    $name `->` type($value) attr-dict
  }];
}

def IREEPyDM_StoreFreeVarUnboxedOp : IREEPyDM_Op<"store_free_var_unboxed"> {
  let summary = "Stores an unboxed free variable";
  let description = [{
    Stores an unboxed value to a free variable slot.
  }];

  let arguments = (ins StrAttr:$name, IREEPyDM_PrimitiveType:$value);

  let assemblyFormat = [{
    $name `,` $value `:` type($value) attr-dict
  }];
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
      FloatType -> FloatAttr<double>
      StrType -> StringAttr
      BytesType -> BytesAttr
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
}

def IREEPyDM_NoneOp : IREEPyDM_PureOp<"none"> {
  let summary = "Gets the singleton NoneType primitive value";
  let results = (outs IREEPyDM_NoneType:$value);
  let assemblyFormat = [{
    `->` type($value) attr-dict
  }];
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
    $value `:` type($value) `->` type(results) attr-dict
  }];
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

//===----------------------------------------------------------------------===//
// Computation
//===----------------------------------------------------------------------===//

def IREEPyDM_DynamicBinaryPromoteOp : IREEPyDM_PureOp<"dynamic_binary_promote"> {
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
}

def IREEPyDM_ApplyBinaryOp : IREEPyDM_PureOp<"apply_binary"> {
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
}

def IREEPyDM_ApplyCompareOp : IREEPyDM_PureOp<"apply_compare"> {
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
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_OPS_TD
