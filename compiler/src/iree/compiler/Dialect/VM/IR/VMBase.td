// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VM_BASE
#define IREE_DIALECT_VM_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE VM (Virtual Machine) dialect
//===----------------------------------------------------------------------===//

def VM_Dialect : Dialect {
  let name = "vm";
  let cppNamespace = "::mlir::iree_compiler::IREE::VM";

  let summary = [{
    A dialect representing operations against an abstract virtual machine.
  }];
  let description = [{
    The virtual machine ops are designed to be either serialized to a bytecode
    representation that can be interpreted at runtime or lowered further to
    static representations such as LLVM IR, C, etc. The idea is that the types
    and operations performed are generally just encoding resource ownership
    rules and control flow that can be represented in many different ways by
    target runtimes. For example, it should be possible to lower the VM dialect
    to SPIR-V and run the VM entirely within a persistent Vulkan kernel.

    With this scalable runtime approach we make some limiting assumptions to
    keep the required implementations simple. As we assume all real math is
    happening within dispatch regions the only math we provide is scalar
    operations used for offset and shape calculations. This also enables simple
    flow control such as fixed-range loops.

    Besides integer values the only other storage type is a variant reference
    modeling an abstract iree_vm_ref_t. This allows automated reference counting
    to be relied upon by other dialects built on top of the VM dialect and
    avoids the need for more verbose manual reference counting logic (that may
    be difficult or impossible to manage given the coroutine-like nature of the
    VM). Lowering targets can insert the reference counting as needed.

    The types in the VM dialect correspond to the storage rather than value
    type, with the interpretation of the type encoded on the op.
  }];
}

//===----------------------------------------------------------------------===//
// Opcode tables
//===----------------------------------------------------------------------===//

// Mandatory:
include "iree/compiler/Dialect/VM/IR/VMOpcodesCore.td"
// Optional:
include "iree/compiler/Dialect/VM/IR/VMOpcodesF32.td"
// Optional:
include "iree/compiler/Dialect/VM/IR/VMOpcodesF64.td"

//===----------------------------------------------------------------------===//
// Declarative encoding framework
//===----------------------------------------------------------------------===//
// These encoding rules are processed by the iree-tblgen gen-vm-op-encoding-defs
// generator to produce the op serialization routines.
//
// The signature of the routine is defined on the VM_SerializableOpInterface as:
//   LogicalResult encode(SymbolTable &syms, VMFuncEncoder &e);
//
// The provided symbol table in 'syms' is all of the module symbols in the
// parent of the active operation. 'e' is the bytecode encoder that the
// operation should be encoded to.
//
// All rules are splatted into the encode function body and are expected to
// return a LogicalResult.
//
// Example declaration:
//   let encoding = [
//     VM_EncOpcode<VM_OPC_Return>,
//     VM_EncVariadicOperands<"operands">
//   ];
//
// Example encode function (pseudo-code, may differ):
//   LogicalResult encode(SymbolTable &syms, VMFuncEncoder &e) {
//     if (failed(e.encodeI8(234)) ||
//         failed(e.encodeOperands(operands()))) {
//       return failure();
//     }
//     return success();
//   }

class VM_EncEncodeExpr<code evalExpr, list<string> parameters = []> {
  // A code snippet that can potentially contain placeholders (e.g., `{0}`).
  code expr = evalExpr;
  // If so, `params` are expected to provide the substitutions for these
  // placeholders. TableGen backends can process the parameters before
  // plugging them in though.
  list<string> params = parameters;
}

class VM_EncConstI8<int value> : VM_EncEncodeExpr<
    "e.encodeI8(" # value # ")">;
class VM_EncOpcode<VM_OPC thisOpcode> : VM_EncEncodeExpr<
    "e.encodeOpcode(\"" # thisOpcode.symbol # "\", " # thisOpcode.value # ")"> {
  VM_OPC opcode = thisOpcode;
}
class VM_EncFuncAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, {0}())", [name]>;
class VM_EncGlobalAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, {0}())", [name]>;
class VM_EncRodataAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, {0}())", [name]>;
class VM_EncType<string expr> : VM_EncEncodeExpr<
    "e.encodeType(" # expr # ")">;
class VM_EncTypeOf<string name> : VM_EncEncodeExpr<
    "e.encodeType({0}())", [name]>;
class VM_EncPrimitiveAttr<string name, int thisBitwidth> : VM_EncEncodeExpr<
    "e.encodePrimitiveAttr(getOperation()->getAttrOfType<TypedAttr>(\"" # name # "\"))"> {
  int bitwidth = thisBitwidth;
}
class VM_EncPrimitiveArrayAttr<string name, int thisBitwidth> : VM_EncEncodeExpr<
    "e.encodePrimitiveArrayAttr(getOperation()->getAttrOfType<DenseElementsAttr>(\"" # name # "\"))"> {
  int bitwidth = thisBitwidth;
}
class VM_EncStrAttr<string name> : VM_EncEncodeExpr<
    "e.encodeStrAttr(getOperation()->getAttrOfType<StringAttr>(\"" # name # "\"))">;
class VM_EncBranch<string blockName, string operandsName, int successorIndex> : VM_EncEncodeExpr<
    "e.encodeBranch({0}(), " # operandsName # "(), " # successorIndex # ")", [blockName]>;
class VM_EncBranchTable<string caseBlockNames, string caseOperandsName, int baseSuccessorIndex> : VM_EncEncodeExpr<
    "e.encodeBranchTable(" # caseBlockNames # "(), " # caseOperandsName # "(), " # baseSuccessorIndex # ")">;
class VM_EncOperand<string name, int ordinal> : VM_EncEncodeExpr<
    "e.encodeOperand({0}(), " # ordinal # ")", [name]>;
class VM_EncVariadicOperands<string name> : VM_EncEncodeExpr<
    "e.encodeOperands({0}())", [name]>;
class VM_EncResult<string name> : VM_EncEncodeExpr<
    "e.encodeResult({0}())", [name]>;
class VM_EncVariadicResults<string name> : VM_EncEncodeExpr<
    "e.encodeResults({0}())", [name]>;

def VM_SerializableOpInterface : OpInterface<"VMSerializableOp"> {
  let description = [{
    Interface used for bytecode VM ops to provide consistent serialization and
    deserialization information.
  }];

  let methods = [
    InterfaceMethod<
      "Encodes the op with the provided bytecode encoder.",
      "LogicalResult",
      "encode",
      (ins "SymbolTable &":$syms, "VMFuncEncoder &":$e)
    >,
    // TODO(benvanik): decoder as a static builder method.
  ];
}

//===----------------------------------------------------------------------===//
// Base VM op classes
//===----------------------------------------------------------------------===//

def VM_OpInterface : OpInterface<"VMOp"> {
  let description = [{
    Interface used for VM ops to provide consistent ordinal assignment and
    register allocation.
  }];

  let methods = [];
}

def VM_ImportInterface : OpInterface<"ImportInterface"> {
  let description = [{
    Interface used for ops in other dialects to expose additional ABI
    information for better marshaling and versioning.

    Ops that implement this interface will be lowered to generic VM import
    declarations and matching vm.call ops during conversion. Ops that do not
    implement this (or are unknown) will be loosely defined based on usage only.
  }];

  let methods = [
    // TODO(benvanik): query ABI information.
    // TODO(benvanik): call builder conversion method?
    // TODO(benvanik): module name override ability? (not just dialect name).

    InterfaceMethod<[{
      Returns true if the import has side-effects (or may have them).

      Implementers should also return true if any arguments are modified
      in-place (such as output buffers being passed in). Returning false will
      allow the op to be removed entirely during optimization and must be
      guaranteed to have no impact on the program if its return values are
      unused.
    }],
    "bool", "hasSideEffects", (ins)>,
  ];
}

class VM_Op<string mnemonic, list<Trait> traits = []> :
    Op<VM_Dialect, mnemonic,
       !listconcat(traits, [VM_OpInterface])> {
  let hasCustomAssemblyFormat = 1;

  list<VM_EncEncodeExpr> encoding = ?;
}

//===----------------------------------------------------------------------===//
// VM traits
//===----------------------------------------------------------------------===//

// Operations with this trait are pseudo-ops that must be converted to other ops
// prior to serialization. There are no runtime equivalents of the op.
def VM_PseudoOp : NativeOpTrait<"IREE::VM::PseudoOp">;

// Operations with this trait are assignment ops that are not serialized into
// bytecode but do survive to keep valid MLIR typing up until serialization.
// Ops must take and return the same number of operands/results but their types
// may differ. During serialization all operand registers will be directly
// reused for all result registers.
def VM_AssignmentOp : NativeOpTrait<"IREE::VM::AssignmentOp">;

// Operations with this trait will be stripped in builds where
// debugging-related operations are not needed (such as release builds).
def VM_DebugOnly : NativeOpTrait<"IREE::VM::DebugOnly">;

// Operations with this trait will disallow code motion across them to ensure
// that execution order before and after the barrier op remains the same.
def VM_FullBarrier : NativeOpTrait<"IREE::VM::FullBarrier">;

// Operations with this trait require the VM f32 extension.
def VM_ExtF32 : NativeOpTrait<"IREE::VM::ExtF32">;
// Operations with this trait require the VM f64 extension.
def VM_ExtF64 : NativeOpTrait<"IREE::VM::ExtF64">;

//===----------------------------------------------------------------------===//
// ref<T> types
//===----------------------------------------------------------------------===//

def VM_AnyRefObject : DialectType<
    Util_Dialect,
    CPred<"IREE::VM::RefType::isCompatible($_self)">,
    "ref_object"> {
  let description = [{
    An object that can be referenced by a vm.ref.
  }];
}

def VM_AnyRef : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::VM::RefType>()">,
    "ref"> {
  let description = [{
    An iree_vm_ref_t/iree::vm::ref<T> of some type.
  }];
}

class VM_RefOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::VM::RefType>()">,
      SubstLeaves<"$_self", "$_self.cast<IREE::VM::RefType>().getObjectType()",
                  type.predicate>
    ]>, "ref<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::VM::RefType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// Buffer types
//===----------------------------------------------------------------------===//

def VM_BufferType : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::VM::BufferType>()">,
    "buffer"> {
  let description = [{
    A byte buffer.
  }];
  let builderCall = "$_builder.getType<IREE::VM::BufferType>()";
}

//===----------------------------------------------------------------------===//
// List types
//===----------------------------------------------------------------------===//

def VM_AnyList : DialectType<
    Util_Dialect,
    And<[
      CPred<"$_self.isa<IREE::VM::RefType>()">,
      CPred<"$_self.cast<IREE::VM::RefType>().getObjectType().isa<IREE::VM::ListType>()">,
    ]>,
    "list"> {
  let description = [{
    An iree_vm_list_t/iree::vm::list<T> of some type.
  }];
}

class VM_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.cast<IREE::VM::RefType>().getObjectType().isa<IREE::VM::ListType>()">,
      Or<[
        CPred<"$_self.cast<IREE::VM::RefType>().getObjectType().cast<IREE::VM::ListType>().getElementType().isa<IREE::VM::OpaqueType>()">,
        SubstLeaves<"$_self",
                    "$_self.cast<IREE::VM::RefType>().getObjectType().cast<IREE::VM::ListType>().getElementType()",
                    type.predicate>
      ]>,
    ]>, "list<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::VM::ListType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// VM types
//===----------------------------------------------------------------------===//

def VM_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def VM_FuncRefAttr : Util_AliasedSymbolRefAttr;

def VM_Ordinal : SignlessIntegerAttrBase<I32, "ordinal value">;

// NOTE: today we only support a single bit width for pointers/indices/sizes.
// This is to keep our compatability matrix smaller but is not a fundamental
// limitation of the VM.
// TODO(#5732): allow for this to change and be specified as an attribute of the
// compiled module (a module compiled with 32-bit indices may not be usable on
// a runtime compiled with 64-bit indices).
def VM_Ptr : I<32>;
def VM_Index : I<32>;
def VM_BufferIndex : I<64>;
def VM_ListIndex : I<32>;

def VM_CondValue : I<32> {
  let description = [{
    Value used to represent boolean conditions where `value == 0` is false and
    `value != 0` is true.
  }];
}

def VM_AnyType : AnyTypeOf<[
  I32,
  I64,
  F32,
  F64,
  VM_CondValue,
  VM_AnyRef,
]>;

def VM_PrimitiveType : AnyTypeOf<[
  AnyIntOfWidths<[8, 16, 32, 64]>,
  FloatOfWidths<[16, 32, 64]>,
]>;

class VM_ConstantIntegerValueAttr<I type> : SignlessIntegerAttrBase<type,
                              type.bitwidth # "-bit integer value"> {
  let storageType = "IntegerAttr";
  let returnType = "IntegerAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

class VM_ConstantFloatValueAttr<F type> :
    FloatAttrBase<type, type.bitwidth # "-bit floating-point value"> {
  let storageType = "FloatAttr";
  let returnType = "FloatAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

//===----------------------------------------------------------------------===//
// VM structs
//===----------------------------------------------------------------------===//

def VM_OrdinalCountsAttr : AttrDef<VM_Dialect, "OrdinalCounts"> {
  let mnemonic = "ordinal_counts";
  let parameters = (ins
    AttrParameter<"int32_t", "">:$import_funcs,
    AttrParameter<"int32_t", "">:$export_funcs,
    AttrParameter<"int32_t", "">:$internal_funcs,
    AttrParameter<"int32_t", "">:$global_bytes,
    AttrParameter<"int32_t", "">:$global_refs,
    AttrParameter<"int32_t", "">:$rodatas,
    AttrParameter<"int32_t", "">:$rwdatas
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // IREE_DIALECT_VM_BASE
