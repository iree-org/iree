// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_INTERFACES
#define IREE_DIALECT_UTIL_IR_UTIL_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE::Util::ClosureOpInterface
//===----------------------------------------------------------------------===//

def Util_ClosureOpInterface : OpInterface<"ClosureOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface for ops that follow the util dialect closure semantics (explicit
    captures, dynamic-shape awareness, and normal operand/result SSA behavior).

    Implementing this interface enables optimizations that perform manipulation
    across the closure capture boundary (outside of the op <-> regions within
    the op).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the body region of the closure (may have multiple blocks).
      }],
      /*retTy=*/"Region &",
      /*methodName=*/"getClosureBodyRegion",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return this->getOperation()->getRegion(0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Returns all closure operand values.}],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getClosureOperands",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns all closure result values.}],
      /*retTy=*/"Operation::result_range",
      /*methodName=*/"getClosureResults",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the given operation can exist in the closure.
        Not all operations that a closure can contain are guaranteed to be folded
        into the closure, such as when the operation may have side-effects.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canClosureContainOp",
      /*args=*/(ins "Operation *":$op)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clones the op while removing specified operands and results.
        The body of the op will be transferred to the new op and the entry block
        will have its arguments removed.

        The returned op will be free standing. Callers must insert it into a block
        where desired (most often just replacing the current op).
      }],
      /*retTy=*/"IREE::Util::ClosureOpInterface",
      /*methodName=*/"cloneReplacementExcludingOperandsAndResults",
      /*args=*/(ins "ArrayRef<unsigned>":$excludedOperandIndices,
                    "ArrayRef<unsigned>":$excludedResultIndices,
                    "PatternRewriter &":$rewriter)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a bitfield indicating how an operand is used within the closure.
      }],
      /*retTy=*/"IREE::Util::ValueAccess",
      /*methodName=*/"getOperandAccess",
      /*args=*/(ins "unsigned":$operandIndex)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a bitfield indicating how a result is used within the closure.
      }],
      /*retTy=*/"IREE::Util::ValueAccess",
      /*methodName=*/"getResultAccess",
      /*args=*/(ins "unsigned":$resultIndex)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::InitializerOpInterface
//===----------------------------------------------------------------------===//

def Util_InitializerOpInterface : OpInterface<"InitializerOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface for ops that behave as module-wide initializers.
    Initializers run in the order they are defined in the module as if they were
    concatenated. All initializers run to completion when modules are
    instantiated though arbitrary globals and functions may be called as part
    of initialization.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the body region of the initializer (may have multiple blocks).
      }],
      /*retTy=*/"Region &",
      /*methodName=*/"getInitializerRegion",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return this->getOperation()->getRegion(0);
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::Global*OpInterface[]
//===----------------------------------------------------------------------===//

def Util_GlobalOpInterface : OpInterface<"GlobalOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface for ops that define global values.
    Globals themselves have value-type semantics but the values stored may not,
    such as if the global is a mutable !util.buffer.

    Globals that are only ever initialized by their attached initial value or
    by stores in an initializer are considered immutable. Loads of immutable
    globals outside of initializers are able to be hoisted and CSEd.

    By default this interface assumes that the global has the following attrs:
    ```
      let arguments = (ins
        // From Symbol:
        OptionalAttr<StrAttr>:$sym_visibility,
        SymbolNameAttr:$sym_name,
        // Assumed by accessors:
        TypeAttr:$type,
        UnitAttr:$is_mutable,
        OptionalAttr<AnyAttr>:$initial_value
      );
    ```
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol name.
      }],
      /*retTy=*/"StringAttr",
      /*methodName=*/"getGlobalName",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getSymNameAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the global is private to the module.
        This is equivalent to SymbolOpInterface::isPrivate but with less nasty
        casting required.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isGlobalPrivate",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return cast<mlir::SymbolOpInterface>($_op.getOperation()).isPrivate();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the type of the global.
      }],
      /*retTy=*/"Type",
      /*methodName=*/"getGlobalType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the type of the global.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalType",
      /*args=*/(ins "Type":$value),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        $_op.setTypeAttr(TypeAttr::get(value));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the global is mutable outside of initializers. Even if
        immutable initializers can store to the global and may do so multiple
        times.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isGlobalMutable",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getIsMutable();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets whether the global can be mutated outside of initializers.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalMutable",
      /*args=*/(ins "bool":$value),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        if (value) {
          $_op.setIsMutableAttr(UnitAttr::get($_op.getContext()));
        } else {
          $_op.removeIsMutableAttr();
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an optional initial value for the global.
        This may differ from the global type if type conversion is supported via
        the GlobalTypeInterface.
      }],
      /*retTy=*/"Attribute",
      /*methodName=*/"getGlobalInitialValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getInitialValueAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets (or clears) the initial value of the global.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalInitialValue",
      /*args=*/(ins "Attribute":$value),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        if (value) {
          // Not fantastic but the typed attr may not accept Attribute.
          $_op->setAttr("initial_value", (value));
        } else {
          $_op.removeInitialValueAttr();
        }
      }]
    >,
  ];

  let verify = [{
    return IREE::Util::detail::verifyGlobalOp(cast<GlobalOpInterface>($_op));
  }];
}

class Util_GlobalAccessorOpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let extraClassDeclaration = [{
    // Returns the global name accessed by the op.
    StringRef getGlobalName() {
      return getGlobalAttr().getValue();
    }
  }];
}

def Util_GlobalAddressOpInterface : Util_GlobalAccessorOpInterface<"GlobalAddressOpInterface"> {
  let description = [{
    Interface for ops that take the address of a global symbol.
    The returned address can be accessed indirectly.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"FlatSymbolRefAttr",
      /*methodName=*/"getGlobalAttr",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalAttr",
      /*args=*/(ins "FlatSymbolRefAttr":$value)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the result of the address operation.
      }],
      /*retTy=*/"OpResult",
      /*methodName=*/"getReturnedGlobalRef",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
    >,
  ];
}

def Util_GlobalLoadOpInterface : Util_GlobalAccessorOpInterface<"GlobalLoadOpInterface"> {
  let description = [{
    Interface for ops that directly load globals via a symbol ref.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"FlatSymbolRefAttr",
      /*methodName=*/"getGlobalAttr",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalAttr",
      /*args=*/(ins "FlatSymbolRefAttr":$value)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the result of the load operation.
      }],
      /*retTy=*/"OpResult",
      /*methodName=*/"getLoadedGlobalValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
    >,
  ];
}

def Util_GlobalStoreOpInterface : Util_GlobalAccessorOpInterface<"GlobalStoreOpInterface"> {
  let description = [{
    Interface for ops that directly stores globals via a symbol ref.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"FlatSymbolRefAttr",
      /*methodName=*/"getGlobalAttr",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setGlobalAttr",
      /*args=*/(ins "FlatSymbolRefAttr":$value)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the value stored by the operation.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getStoredGlobalValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getValue();
      }]
    >,
  ];
}

class Util_GlobalIndirectAccessorOpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";
}

def Util_GlobalLoadIndirectOpInterface : Util_GlobalIndirectAccessorOpInterface<"GlobalLoadIndirectOpInterface"> {
  let description = [{
    Interface for ops that indirectly load globals via a pointer.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getGlobal",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the result of the load operation.
      }],
      /*retTy=*/"OpResult",
      /*methodName=*/"getLoadedGlobalValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
    >,
  ];
}

def Util_GlobalStoreIndirectOpInterface : Util_GlobalIndirectAccessorOpInterface<"GlobalStoreIndirectOpInterface"> {
  let description = [{
    Interface for ops that indirect stores globals via a pointer.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the global symbol accessed by the op.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getGlobal",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the value stored by the operation.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getStoredGlobalValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getValue();
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::NumericCastOpInterface
//===----------------------------------------------------------------------===//

def Util_NumericCastOpInterface : OpInterface<"NumericCastOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Applied to numeric casting ops which can convert between different numeric
    types or shaped-types thereof. Example ops include `fptosi`, `trunci`, etc.
    Treating these generically allows us to perform various cast movement
    optimizations.

    Conforming operations must:
      * Have no attributes.
      * Have one operand and one result.
      * Be able to operate on supported scalar types: IntegerType, FloatType,
        IndexType.
      * Be able to operate on tensors/vectors of supported scalar types.
      * Have a builder that takes (Type, Value).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Gets the input value.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getInput",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Gets the input type.
      }],
      /*retTy=*/"Type",
      /*methodName=*/"getInputType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getOperand(0).getType();
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Gets the result casted value.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getCasted",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the result casted type.
      }],
      /*retTy=*/"Type",
      /*methodName=*/"getCastedType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0).getType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clones the operation with a new result type and input.
        Note that it is generally not legal to generically change the scalar
        type of the input or the result, but it is legal to transform between
        a scalar type and a tensor/vector with an element type of the original
        scalar type (and vica-versa).
      }],
      /*retTy=*/"NumericCastOpInterface",
      /*methodName=*/"cloneWithInput",
      /*args=*/(ins "OpBuilder &":$builder, "Type":$resultType, "Value":$input),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
          return llvm::cast<NumericCastOpInterface>(
            builder.create<ConcreteOp>($_op->getLoc(), resultType, input)
              .getOperation());
      }]
    >,
  ];
}


//===----------------------------------------------------------------------===//
// IREE::Util::TiedOpInterface
//===----------------------------------------------------------------------===//

def Util_TiedOpInterface : OpInterface<"TiedOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    An operation that "ties" one or more results to its operands indicating
    that the result is directly related to the operand in an operation-defined
    way. Results are still SSA values distinct from the operands and the tie is
    strictly a relationship relevant to transformations and not something that
    modifies IR definitions.

    Example:
      An operation on tensors that wants to indicate that the storage for a
      result should alias the storage for an operand, performing an "in-place"
      operation. Since tensors are still used there is no hard requirement that
      uses of the result SSA value alias the operand; a copy may still be
      introduced.

      See: flow.dispatch.workgroups

    Example:
      An operation on buffers that wants to encode activity on the buffer in IR
      (such as a barrier, a transfer operation, etc) such that the SSA use-def
      chain is representing the state of the buffer at various points in time
      but that the underlying buffers are all tied together.

      See: hal.stream.barrier

    The default implementations use an attribute on the op to store the
    relationship:
      `OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands`

    Note that `$tied_operands` are indices inside the operand range returned
    by `getTiedOperandsIndexAndLength`, which may *not* be the full operand
    range of the op.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the set of operands that results may be tied to as an
        (index, length) pair ala getODSOperandIndexAndLength.

        By default assumes all operands may be tied. If an op treats some
        operands as special then the op can override this and specify only the
        ones it will tie. For example, a cond_branch that has a condition
        operand as well as the successor operands would return only the range
        of successor operands.
      }],
      /*retTy=*/"std::pair<unsigned, unsigned>",
      /*methodName=*/"getTiedOperandsIndexAndLength", (ins),
      /*args=*/[{}],
      /*defaultImplementation=*/[{
        return {0, $_op->getNumOperands()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the set of results that may be tied to operands as an
        (index, length) pair ala getODSResultIndexAndLength.

        By default assumes all results may be tied. If an op treats some
        results as special then the op can override this and specify only the
        ones it will tie.
      }],
      /*retTy=*/"std::pair<unsigned, unsigned>",
      /*methodName=*/"getTiedResultsIndexAndLength", (ins),
      /*args=*/[{}],
      /*defaultImplementation=*/[{
        return {0, $_op->getNumResults()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Walks up the SSA use-def chain to find the first defined value reachable
        from the given value by traversing tied ops. The returned value may be
        in another block if that block dominates the one the result is defined
        in.

        Note that the returned value may be a block argument and have no
        defining op, and the search will not continue past branches.
        If the result is untied then the result itself is returned.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getTiedResult",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::TiedOpInterface::findTiedBaseValue($_op.getResult(resultIndex));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand tied to the given result of the op or nullptr if
        none.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getTiedResultOperand",
      /*args=*/(ins "Value":$result),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto resultIndex = result.cast<mlir::OpResult>().getResultNumber();
        auto operandIndex = cast<TiedOpInterface>($_op.getOperation())
            .getTiedResultOperandIndex(resultIndex);
        return operandIndex.has_value() ?
            $_op.getOperand(operandIndex.value()) :
            nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand index tied to the given result index, if any.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"::std::optional<unsigned>",
      /*methodName=*/"getTiedResultOperandIndex",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::detail::getTiedResultOperandIndex($_op, resultIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the operand index tied to the given result index, if any.

        Note that the index should be into the operand range returned by
        `getTiedOperandsIndexAndLength`.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setTiedResultOperandIndex",
      /*args=*/(ins "unsigned":$resultIndex,
                    "::std::optional<unsigned>":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::detail::setTiedResultOperandIndex($_op, resultIndex, operandIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an array containing the tied result operand indices with -1
        indicating that a result is not tied.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getTiedResultOperandIndices",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::detail::getTiedResultOperandIndices($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the given flattened operand index is tied to one or more
        results.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOperandTied",
      /*args=*/(ins "unsigned":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::detail::isOperandTied($_op, operandIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a list of result values that are tied to the given operand.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getOperandTiedResults",
      /*args=*/(ins "unsigned":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Util::detail::getOperandTiedResults($_op, operandIndex);
      }]
    >,
  ];

  let extraClassDeclaration = [{
    static StringRef getStorageAttrName() { return "tied_operands"; }

    // Indicates that a result is not tied to any operand.
    static constexpr int64_t kUntiedIndex = -1;

    // Walks the SSA use-def chain to find the first defined value reachable
    // from the given value by traversing tied ops. Note that the returned
    // value may be a block argument and have no defining op.
    static Value findTiedBaseValue(Value derivedValue);

    // Returns true if any of |value|'s uses have tied it to a result.
    static bool hasAnyTiedUses(Value value);
  }];

  let verify = [{
    return IREE::Util::detail::verifyTiedOp(cast<TiedOpInterface>($_op));
  }];
}

//===----------------------------------------------------------------------===//
// IREE::Util::ShapeAware* interfaces
//===----------------------------------------------------------------------===//

def Util_ShapeAwareOp : OpInterface<"ShapeAwareOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    An operation that is able to provide dynamic shape dimensions for all
    shaped operands and results.

    This is a more fine-grained version of ReifyRankedShapedTypeOpInterface that
    allows for querying of individual operand and result dimensions without
    unconditionally inserting IR. The dynamic dimension queries allow us to find
    the dynamic shape dimension SSA values in the IR in read-only mode (such as
    during analysis), and having the queries specific to a particular operand
    and result allows us to walk through ops along use-def edges. When combined
    with tied operands the split queries allow for operands and results to have
    differing dimensions (such as after reshaping/casting).

    `getOperandDynamicDims` and `getResultDynamicDims` are the load-bearing
    methods and if there was an equivalent set in an upstream dialect we could
    switch to using that instead as the rest are just utilities built around
    them.

    `tensor.dim` and `memref.dim` will both resolve shape dimensions through
    this interface when the dimension index is constant.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns dynamic dimensions for the given shaped operand index.}],
      "mlir::ValueRange", "getOperandDynamicDims", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns dynamic dimensions for the given shaped result index.}],
      "mlir::ValueRange", "getResultDynamicDims", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns dynamic dimensions for the given shaped result value.}],
      "mlir::ValueRange", "getResultDynamicDimsFromValue", (ins "Value":$value),
      /*defaultImplementation=*/[{
        for (unsigned i = 0; i < $_self->getNumResults(); ++i) {
          if ($_self->getResult(i) == value) {
            return $_self.getResultDynamicDims(i);
          }
        }
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Returns a full shape for the given shaped operand index.}],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"buildOperandShape",
      /*args=*/(ins "unsigned":$idx, "OpBuilder &":$builder),
      /*defaultImplementation=*/[{
        return IREE::Util::buildOperandShape($_self, idx, builder);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Returns a full shape for the given shaped result index.}],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"buildResultShape",
      /*args=*/(ins "unsigned":$idx, "OpBuilder &":$builder),
      /*defaultImplementation=*/[{
        return IREE::Util::buildResultShape($_self, idx, builder);
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/[{Builds a full shape for the given shaped result value.}],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"buildResultValueShape",
      /*args=*/(ins "Value":$result, "OpBuilder &":$builder),
      /*methodBody=*/[{
        auto shapeAwareOp = dyn_cast<IREE::Util::ShapeAwareOpInterface>(result.getDefiningOp());
        return shapeAwareOp.buildResultShape(
            result.cast<mlir::OpResult>().getResultNumber(), builder);
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::SizeAware* interfaces
//===----------------------------------------------------------------------===//

def Util_InferTypeSize : TypeInterface<"InferTypeSizeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Allows types to be queried for their size by inserting the required logic.
  }];

  let methods = [
    InterfaceMethod<
      [{Builds an expression computing the size of the value.}],
      "Value", "inferSizeFromValue", (ins "Location":$loc,
                                          "Value":$value,
                                          "OpBuilder &":$builder)
    >,
  ];
}

def Util_SizeAwareType : TypeInterface<"SizeAwareTypeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Denotes that a type is size-aware and must always have a size value
    associated with it in the IR. See `SizeAwareOp` for more information.
  }];

  let extraClassDeclaration = [{
    // Walks the SSA use-def chain to find the size of the type.
    // Returns nullptr if the size cannot be found or if it is defined after
    // {|block|, |insertionPoint|}.
    static Value findSizeValue(Value resourceValue, Block *block,
                               Block::iterator insertionPoint);

    // Returns an SSA value representing the byte size of |value| or nullptr
    // if not a sized value.
    static Value queryValueSize(Location loc, Value resourceValue,
                                OpBuilder &builder);
  }];
}

def Util_SizeAwareOp : OpInterface<"SizeAwareOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    An operation that is able to provide size values for all size-aware operands
    and results.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns a size for the given sized operand index.}],
      "Value", "getOperandSize", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns a size for the given sized result index.}],
      "Value", "getResultSize", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns a size for the given sized result value.}],
      "Value", "getResultSizeFromValue", (ins "Value":$value),
      /*defaultImplementation=*/[{
        for (unsigned i = 0; i < $_self->getNumResults(); ++i) {
          if ($_self->getResult(i) == value) return $_self.getResultSize(i);
        }
        return {};
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::ReferenceTypeInterface
//===----------------------------------------------------------------------===//

def Util_ReferenceType : TypeInterface<"ReferenceTypeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface used to mark types as passed by-reference. These types may have
    aliases/ties or be convertable into forms that do. This is in contrast to
    the default of immutable by-value types.
  }];
}

//===----------------------------------------------------------------------===//
// IREE::Util::GlobalTypeInterface
//===----------------------------------------------------------------------===//

def Util_GlobalType : TypeInterface<"GlobalTypeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface used to mark types as supported for global variables.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the access to a global via a specific type is compatible
        with the storage type of the global. For example, a storage of
        `tensor<?xf32>` is compatible with an access as `tensor<4xf32>`.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isAccessStorageCompatible",
      /*args=*/(ins "Type":$accessType),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // If one is a shaped type, then they both must be and have compatible
        // shapes.
        if ($_type.template isa<ShapedType>() || accessType.isa<ShapedType>()) {
          return succeeded(mlir::verifyCompatibleShape($_type, accessType));
        }
        // Otherwise, the types must be the same.
        return $_type == accessType;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::Subrange*Interface
//===----------------------------------------------------------------------===//

def Util_SubrangeType : TypeInterface<"SubrangeTypeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface used on size-aware types that can represent linear subranges as an
    (offset, length).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Creates an op returning an (offset, length) subrange of a resource.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"createSubrangeOp",
      /*args=*/(ins
        "Location":$loc,
        "Value":$resource,
        "Value":$resourceSize,
        "Value":$subrangeOffset,
        "Value":$subrangeLength,
        "OpBuilder &":$builder
      ),
      /*methodBody=*/[{}]
    >,
  ];
}

def Util_SubrangeOp : OpInterface<"SubrangeOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    A size-aware operation taking a subrange on a resource.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the size-aware source resource the subrange is taken from.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getSubrangeResource",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the total size of the source resource.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getSubrangeResourceSize",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the offset of the taken subrange into the resource.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getSubrangeOffset",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the length of the taken subrange.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getSubrangeLength",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the resulting subrange value if the op produces one.
        Will be nullptr if the op is only consuming a subrange.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getSubrangeResult",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
  ];
}

def Util_SubrangeOperandOpInterface : OpInterface<"SubrangeOperandOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface for operations that consume subranges of size-aware resources.
    The methods are used to manipulate the subranges during transformation.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the subrange operand values for the given flat operand index.
      }],
      /*retTy=*/"IREE::Util::SubrangeOperand",
      /*methodName=*/"getSubrangeOperand",
      /*args=*/(ins "unsigned":$operandIndex)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Updates the subrange operand values for the given flat operand index.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setSubrangeOperand",
      /*args=*/(ins
        "unsigned":$operandIndex,
        "IREE::Util::SubrangeOperand":$operand
      )
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Util::SerializableAttrInterface
//===----------------------------------------------------------------------===//

def Util_SerializableAttrInterface : AttrInterface<"SerializableAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let description = [{
    Interface used to serialize constant values in binary format.
    Implementations are responsible for encoding and may do so in any form they
    desire. For example, an array of dense float32 elements may be written
    directly as the elements while sparse float32 elements may have a sparsity
    header in a defined format.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the storage size in bytes required by the serialized value.
        Any of the serialization methods will write precisely this number of
        bytes. If the value is sub-byte aligned the storage size will be rounded
        up to the next whole byte.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getStorageSize",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Serializes the attribute to the given byte vector. The vector will be
        resized to the total storage size upon return.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"serializeToVector",
      /*args=*/(ins
        "Location":$loc,
        "llvm::endianness":$endian,
        "SmallVectorImpl<char> &":$buffer
      ),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        buffer.resize($_attr.getStorageSize());
        return $_attr.serializeToBuffer(loc, endian, buffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Serializes the attribute to the given byte buffer. The buffer must have
        a capacity of at least that returned by getStorageSize.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"serializeToBuffer",
      /*args=*/(ins
        "Location":$loc,
        "llvm::endianness":$endian,
        "ArrayRef<char>":$buffer
      )
    >,
    InterfaceMethod<
      /*desc=*/[{
        Serializes the attribute to the given output stream. Where possible
        implementations are encouraged to avoid allocations on this path.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"serializeToStream",
      /*args=*/(ins
        "Location":$loc,
        "llvm::endianness":$endian,
        "llvm::raw_ostream &":$os
      ),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        SmallVector<char> buffer;
        if (failed($_attr.serializeToVector(loc, endian, buffer))) {
          return failure();
        }
        os.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
        return success();
      }]
    >,
  ];
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_INTERFACES
