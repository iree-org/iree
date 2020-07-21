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

#ifndef IREE_DIALECT_VM_BASE
#define IREE_DIALECT_VM_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE VM (Virtual Machine) dialect
//===----------------------------------------------------------------------===//

def VM_Dialect : Dialect {
  let name = "vm";
  let cppNamespace = "IREE::VM";

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
    modeling an abstract iree::ref_ptr. This allows automated reference counting
    to be relied upon by other dialects built on top of the VM dialect and
    avoids the need for more verbose manual reference counting logic (that may
    be difficult or impossible to manage given the coroutine-like nature of the
    VM). Lowering targets can insert the reference counting as needed.
  }];
}

//===----------------------------------------------------------------------===//
// VM opcodes
//===----------------------------------------------------------------------===//
// Opcode ranges:
// 0x00-0x9F: core VM opcodes, reserved for this dialect
// 0xA0-0xFF: unreserved, used to prefix extension op sets
//
// Note that changing existing opcode assignments will invalidate all binaries
// and should only be done when breaking changes are acceptable. We could add a
// versioning system here to automatically switch between different encodings
// but we are a long way out to stabilizing this format :)
//
// Some opcodes require an extension prefix to indicate that runtime support
// is optional. An op with the ExtI64 trait will require VM_OPC_ExtI64, for
// example. Ops that bridge extension sets have a canonical form that may
// require multiple prefix codes (for example, the i64<->f64 extensions).

class VM_OPC<int opcode, string name> :
    IntEnumAttrCaseBase<I8, name, name, opcode>;

class VM_OPC_EnumAttr<string name, string enumName, string enumTag,
                      string description,
                      VM_OPC prefix = ?,
                      list<VM_OPC> cases> :
    IntEnumAttr<I8, name, description, cases> {
  let cppNamespace = "IREE::VM";
  let returnType = cppNamespace # "::" # name;
  let underlyingType = "uint8_t";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall =
          "$_builder.getI8IntegerAttr(static_cast<int8_t>($0))";

  // Used by VMOpTableGen:
  string opcodeEnumName = enumName;
  VM_OPC opcodePrefix = prefix;
  string opcodeEnumTag = enumTag;
}

// Globals:
def VM_OPC_GlobalLoadI32         : VM_OPC<0x00, "GlobalLoadI32">;
def VM_OPC_GlobalStoreI32        : VM_OPC<0x01, "GlobalStoreI32">;
def VM_OPC_GlobalLoadIndirectI32 : VM_OPC<0x02, "GlobalLoadIndirectI32">;
def VM_OPC_GlobalStoreIndirectI32: VM_OPC<0x03, "GlobalStoreIndirectI32">;
def VM_OPC_GlobalLoadRef         : VM_OPC<0x04, "GlobalLoadRef">;
def VM_OPC_GlobalStoreRef        : VM_OPC<0x05, "GlobalStoreRef">;
def VM_OPC_GlobalLoadIndirectRef : VM_OPC<0x06, "GlobalLoadIndirectRef">;
def VM_OPC_GlobalStoreIndirectRef: VM_OPC<0x07, "GlobalStoreIndirectRef">;

// Constants:
def VM_OPC_ConstI32Zero          : VM_OPC<0x08, "ConstI32Zero">;
def VM_OPC_ConstI32              : VM_OPC<0x09, "ConstI32">;
def VM_OPC_ConstRefZero          : VM_OPC<0x0A, "ConstRefZero">;
def VM_OPC_ConstRefRodata        : VM_OPC<0x0B, "ConstRefRodata">;

// List operations:
def VM_OPC_ListAlloc             : VM_OPC<0x10, "ListAlloc">;
def VM_OPC_ListReserve           : VM_OPC<0x11, "ListReserve">;
def VM_OPC_ListSize              : VM_OPC<0x12, "ListSize">;
def VM_OPC_ListResize            : VM_OPC<0x13, "ListResize">;
def VM_OPC_ListGetI32            : VM_OPC<0x14, "ListGetI32">;
def VM_OPC_ListSetI32            : VM_OPC<0x15, "ListSetI32">;
def VM_OPC_ListGetRef            : VM_OPC<0x16, "ListGetRef">;
def VM_OPC_ListSetRef            : VM_OPC<0x17, "ListSetRef">;
// RESERVED: 0x18 push.i32
// RESERVED: 0x19 pop.i32
// RESERVED: 0x1A copy to other list
// RESERVED: 0x1B slice clone into new list
// RESERVED: 0x1C read byte buffer?
// RESERVED: 0x1D write byte buffer?

// Conditional assignment:
def VM_OPC_SelectI32             : VM_OPC<0x1E, "SelectI32">;
def VM_OPC_SelectRef             : VM_OPC<0x1F, "SelectRef">;
def VM_OPC_SwitchI32             : VM_OPC<0x20, "SwitchI32">;
def VM_OPC_SwitchRef             : VM_OPC<0x21, "SwitchRef">;

// Integer arithmetic and logic:
def VM_OPC_AddI32                : VM_OPC<0x22, "AddI32">;
def VM_OPC_SubI32                : VM_OPC<0x23, "SubI32">;
def VM_OPC_MulI32                : VM_OPC<0x24, "MulI32">;
def VM_OPC_DivI32S               : VM_OPC<0x25, "DivI32S">;
def VM_OPC_DivI32U               : VM_OPC<0x26, "DivI32U">;
def VM_OPC_RemI32S               : VM_OPC<0x27, "RemI32S">;
def VM_OPC_RemI32U               : VM_OPC<0x28, "RemI32U">;
def VM_OPC_NotI32                : VM_OPC<0x29, "NotI32">;
def VM_OPC_AndI32                : VM_OPC<0x2A, "AndI32">;
def VM_OPC_OrI32                 : VM_OPC<0x2B, "OrI32">;
def VM_OPC_XorI32                : VM_OPC<0x2C, "XorI32">;

// Bitwise shifts and rotates:
def VM_OPC_ShlI32                : VM_OPC<0x2D, "ShlI32">;
def VM_OPC_ShrI32S               : VM_OPC<0x2E, "ShrI32S">;
def VM_OPC_ShrI32U               : VM_OPC<0x2F, "ShrI32U">;

// Casting and type conversion/emulation:
def VM_OPC_TruncI32I8            : VM_OPC<0x31, "TruncI32I8">;
def VM_OPC_TruncI32I16           : VM_OPC<0x32, "TruncI32I16">;
def VM_OPC_ExtI8I32S             : VM_OPC<0x33, "ExtI8I32S">;
def VM_OPC_ExtI8I32U             : VM_OPC<0x34, "ExtI8I32U">;
def VM_OPC_ExtI16I32S            : VM_OPC<0x35, "ExtI16I32S">;
def VM_OPC_ExtI16I32U            : VM_OPC<0x36, "ExtI16I32U">;

// Reduction arithmetic:

// Comparison ops:
def VM_OPC_CmpEQI32              : VM_OPC<0x40, "CmpEQI32">;
def VM_OPC_CmpNEI32              : VM_OPC<0x41, "CmpNEI32">;
def VM_OPC_CmpLTI32S             : VM_OPC<0x42, "CmpLTI32S">;
def VM_OPC_CmpLTI32U             : VM_OPC<0x43, "CmpLTI32U">;
def VM_OPC_CmpNZI32              : VM_OPC<0x4D, "CmpNZI32">;
def VM_OPC_CmpEQRef              : VM_OPC<0x4A, "CmpEQRef">;
def VM_OPC_CmpNERef              : VM_OPC<0x4B, "CmpNERef">;
def VM_OPC_CmpNZRef              : VM_OPC<0x4C, "CmpNZRef">;

// Control flow:
def VM_OPC_Branch                : VM_OPC<0x50, "Branch">;
def VM_OPC_CondBranch            : VM_OPC<0x51, "CondBranch">;
def VM_OPC_Call                  : VM_OPC<0x52, "Call">;
def VM_OPC_CallVariadic          : VM_OPC<0x53, "CallVariadic">;
def VM_OPC_Return                : VM_OPC<0x54, "Return">;
def VM_OPC_Fail                  : VM_OPC<0x55, "Fail">;

// Async/fiber ops:
def VM_OPC_Yield                 : VM_OPC<0x60, "Yield">;

// Debugging:
def VM_OPC_Trace                 : VM_OPC<0x7C, "Trace">;
def VM_OPC_Print                 : VM_OPC<0x7D, "Print">;
def VM_OPC_CondBreak             : VM_OPC<0x7E, "CondBreak">;
def VM_OPC_Break                 : VM_OPC<0x7F, "Break">;

// Extension prefixes:
def VM_OPC_PrefixExtI64          : VM_OPC<0xA0, "PrefixExtI64">;
def VM_OPC_PrefixExtF32          : VM_OPC<0xA1, "PrefixExtF32">;
def VM_OPC_PrefixExtF64          : VM_OPC<0xA2, "PrefixExtF64">;

// Runtime enum iree_vm_core_op_t:
def VM_CoreOpcodeAttr :
    VM_OPC_EnumAttr<"Opcode",
                    "iree_vm_core_op_t",
                    "CORE",  // IREE_VM_OP_CORE_*
                    "valid VM core operation encodings",
                    ?, [
    // Core VM opcodes (0x00-0x9F):
    VM_OPC_GlobalLoadI32,
    VM_OPC_GlobalStoreI32,
    VM_OPC_GlobalLoadIndirectI32,
    VM_OPC_GlobalStoreIndirectI32,
    VM_OPC_GlobalLoadRef,
    VM_OPC_GlobalStoreRef,
    VM_OPC_GlobalLoadIndirectRef,
    VM_OPC_GlobalStoreIndirectRef,
    VM_OPC_ConstI32Zero,
    VM_OPC_ConstI32,
    VM_OPC_ConstRefZero,
    VM_OPC_ConstRefRodata,
    VM_OPC_ListAlloc,
    VM_OPC_ListReserve,
    VM_OPC_ListSize,
    VM_OPC_ListResize,
    VM_OPC_ListGetI32,
    VM_OPC_ListSetI32,
    VM_OPC_ListGetRef,
    VM_OPC_ListSetRef,
    VM_OPC_SelectI32,
    VM_OPC_SelectRef,
    VM_OPC_SwitchI32,
    VM_OPC_SwitchRef,
    VM_OPC_AddI32,
    VM_OPC_SubI32,
    VM_OPC_MulI32,
    VM_OPC_DivI32S,
    VM_OPC_DivI32U,
    VM_OPC_RemI32S,
    VM_OPC_RemI32U,
    VM_OPC_NotI32,
    VM_OPC_AndI32,
    VM_OPC_OrI32,
    VM_OPC_XorI32,
    VM_OPC_ShlI32,
    VM_OPC_ShrI32S,
    VM_OPC_ShrI32U,
    VM_OPC_TruncI32I8,
    VM_OPC_TruncI32I16,
    VM_OPC_ExtI8I32S,
    VM_OPC_ExtI8I32U,
    VM_OPC_ExtI16I32S,
    VM_OPC_ExtI16I32U,
    VM_OPC_CmpEQI32,
    VM_OPC_CmpNEI32,
    VM_OPC_CmpLTI32S,
    VM_OPC_CmpLTI32U,
    VM_OPC_CmpNZI32,
    VM_OPC_CmpEQRef,
    VM_OPC_CmpNERef,
    VM_OPC_CmpNZRef,
    VM_OPC_Branch,
    VM_OPC_CondBranch,
    VM_OPC_Call,
    VM_OPC_CallVariadic,
    VM_OPC_Return,
    VM_OPC_Fail,
    VM_OPC_Yield,
    VM_OPC_Trace,
    VM_OPC_Print,
    VM_OPC_CondBreak,
    VM_OPC_Break,

    // Extension opcodes (0xA0-0xFF):
    VM_OPC_PrefixExtI64,  // VM_ExtI64OpcodeAttr
    VM_OPC_PrefixExtF32,  // VM_ExtF32OpcodeAttr
    VM_OPC_PrefixExtF64,  // VM_ExtF64OpcodeAttr
  ]>;

// i64 extension:
// (ops are encoded as a VM_OPC_ExtI64 + the opcode below)
def VM_OPC_GlobalLoadI64         : VM_OPC<0x00, "GlobalLoadI64">;
def VM_OPC_GlobalStoreI64        : VM_OPC<0x01, "GlobalStoreI64">;
def VM_OPC_GlobalLoadIndirectI64 : VM_OPC<0x02, "GlobalLoadIndirectI64">;
def VM_OPC_GlobalStoreIndirectI64: VM_OPC<0x03, "GlobalStoreIndirectI64">;
def VM_OPC_ConstI64Zero          : VM_OPC<0x08, "ConstI64Zero">;
def VM_OPC_ConstI64              : VM_OPC<0x09, "ConstI64">;
def VM_OPC_ListGetI64            : VM_OPC<0x14, "ListGetI64">;
def VM_OPC_ListSetI64            : VM_OPC<0x15, "ListSetI64">;
def VM_OPC_SelectI64             : VM_OPC<0x1E, "SelectI64">;
def VM_OPC_SwitchI64             : VM_OPC<0x20, "SwitchI64">;
def VM_OPC_AddI64                : VM_OPC<0x22, "AddI64">;
def VM_OPC_SubI64                : VM_OPC<0x23, "SubI64">;
def VM_OPC_MulI64                : VM_OPC<0x24, "MulI64">;
def VM_OPC_DivI64S               : VM_OPC<0x25, "DivI64S">;
def VM_OPC_DivI64U               : VM_OPC<0x26, "DivI64U">;
def VM_OPC_RemI64S               : VM_OPC<0x27, "RemI64S">;
def VM_OPC_RemI64U               : VM_OPC<0x28, "RemI64U">;
def VM_OPC_NotI64                : VM_OPC<0x29, "NotI64">;
def VM_OPC_AndI64                : VM_OPC<0x2A, "AndI64">;
def VM_OPC_OrI64                 : VM_OPC<0x2B, "OrI64">;
def VM_OPC_XorI64                : VM_OPC<0x2C, "XorI64">;
def VM_OPC_ShlI64                : VM_OPC<0x2D, "ShlI64">;
def VM_OPC_ShrI64S               : VM_OPC<0x2E, "ShrI64S">;
def VM_OPC_ShrI64U               : VM_OPC<0x2F, "ShrI64U">;
def VM_OPC_TruncI64I32           : VM_OPC<0x32, "TruncI64I32">;
def VM_OPC_ExtI32I64S            : VM_OPC<0x37, "ExtI32I64S">;
def VM_OPC_ExtI32I64U            : VM_OPC<0x38, "ExtI32I64U">;
def VM_OPC_CmpEQI64              : VM_OPC<0x40, "CmpEQI64">;
def VM_OPC_CmpNEI64              : VM_OPC<0x41, "CmpNEI64">;
def VM_OPC_CmpLTI64S             : VM_OPC<0x42, "CmpLTI64S">;
def VM_OPC_CmpLTI64U             : VM_OPC<0x43, "CmpLTI64U">;
def VM_OPC_CmpNZI64              : VM_OPC<0x4D, "CmpNZI64">;

// Runtime enum iree_vm_ext_i64_op_t:
def VM_ExtI64OpcodeAttr :
    VM_OPC_EnumAttr<"ExtI64Opcode",
                    "iree_vm_ext_i64_op_t",
                    "EXT_I64",  // IREE_VM_OP_EXT_I64_*
                    "valid VM operation encodings in the i64 extension",
                    VM_OPC_PrefixExtI64, [
    VM_OPC_GlobalLoadI64,
    VM_OPC_GlobalStoreI64,
    VM_OPC_GlobalLoadIndirectI64,
    VM_OPC_GlobalStoreIndirectI64,
    VM_OPC_ConstI64Zero,
    VM_OPC_ConstI64,
    VM_OPC_ListGetI64,
    VM_OPC_ListSetI64,
    VM_OPC_SelectI64,
    VM_OPC_SwitchI64,
    VM_OPC_AddI64,
    VM_OPC_SubI64,
    VM_OPC_MulI64,
    VM_OPC_DivI64S,
    VM_OPC_DivI64U,
    VM_OPC_RemI64S,
    VM_OPC_RemI64U,
    VM_OPC_NotI64,
    VM_OPC_AndI64,
    VM_OPC_OrI64,
    VM_OPC_XorI64,
    VM_OPC_ShlI64,
    VM_OPC_ShrI64S,
    VM_OPC_ShrI64U,
    VM_OPC_TruncI64I32,
    VM_OPC_ExtI32I64S,
    VM_OPC_ExtI32I64U,
    VM_OPC_CmpEQI64,
    VM_OPC_CmpNEI64,
    VM_OPC_CmpLTI64S,
    VM_OPC_CmpLTI64U,
    VM_OPC_CmpNZI64,
  ]>;

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

class VM_EncEncodeExpr<code evalExpr> {
  code expr = evalExpr;
}

class VM_EncConstI8<int value> : VM_EncEncodeExpr<
    "e.encodeI8(" # value # ")">;
class VM_EncOpcode<VM_OPC thisOpcode> : VM_EncEncodeExpr<
    "e.encodeOpcode(\"" # thisOpcode.symbol # "\", " # thisOpcode.value # ")"> {
  VM_OPC opcode = thisOpcode;
}
class VM_EncFuncAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, " # name # "())">;
class VM_EncGlobalAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, " # name # "())">;
class VM_EncRodataAttr<string name> : VM_EncEncodeExpr<
    "e.encodeSymbolOrdinal(syms, " # name # "())">;
class VM_EncType<string expr> : VM_EncEncodeExpr<
    "e.encodeType(" # expr # ")">;
class VM_EncTypeOf<string name> : VM_EncEncodeExpr<
    "e.encodeType(" # name # "())">;
class VM_EncIntAttr<string name, int thisBitwidth> : VM_EncEncodeExpr<
    "e.encodeIntAttr(getAttrOfType<IntegerAttr>(\"" # name # "\"))"> {
  int bitwidth = thisBitwidth;
}
class VM_EncIntArrayAttr<string name, int thisBitwidth> : VM_EncEncodeExpr<
    "e.encodeIntArrayAttr(getAttrOfType<DenseIntElementsAttr>(\"" # name # "\"))"> {
  int bitwidth = thisBitwidth;
}
class VM_EncStrAttr<string name> : VM_EncEncodeExpr<
    "e.encodeStrAttr(getAttrOfType<StringAttr>(\"" # name # "\"))">;
class VM_EncBranch<string blockName, string operandsName, int successorIndex> : VM_EncEncodeExpr<
    "e.encodeBranch(" # blockName # "(), " # operandsName # "(), " # successorIndex # ")">;
class VM_EncOperand<string name, int ordinal> : VM_EncEncodeExpr<
    "e.encodeOperand(" # name # "(), " # ordinal # ")">;
class VM_EncVariadicOperands<string name> : VM_EncEncodeExpr<
    "e.encodeOperands(" # name # "())">;
class VM_EncResult<string name> : VM_EncEncodeExpr<
    "e.encodeResult(" # name # "())">;
class VM_EncVariadicResults<string name> : VM_EncEncodeExpr<
    "e.encodeResults(" # name # "())">;

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

class VM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<VM_Dialect, mnemonic,
       !listconcat(traits, [VM_OpInterface])> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];

  list<VM_EncEncodeExpr> encoding = ?;
}

def VM_GlobalOpInterface : OpInterface<"VMGlobalOp"> {
  let description = [{
    Interface used for VM ops that declare global values.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the storage type of this global such as i32.
    }],
    "Type", "getStorageType", (ins)>,
    InterfaceMethod<[{
      Returns the size in bytes of the global when stored in rwdata. Valid only
      for globals using primitive storage.
    }],
    "size_t", "getStorageSize", (ins), [{
      auto storageType = $_self.getStorageType();
      assert(storageType.isIntOrFloat());
      assert(storageType.getIntOrFloatBitWidth() % 8 == 0);
      return storageType.getIntOrFloatBitWidth() / 8;
    }]>,
    InterfaceMethod<[{}], "StringRef", "getSymbolName", (ins)>,
    InterfaceMethod<[{}], "bool", "isMutable", (ins)>,
    InterfaceMethod<[{}], "Optional<StringRef>", "getInitializerAttr", (ins)>,
    InterfaceMethod<[{}], "Optional<Attribute>", "getInitialValueAttr", (ins)>,
    InterfaceMethod<[{}], "Optional<IntegerAttr>", "getOrdinalAttr", (ins)>,
    InterfaceMethod<[{}], "int", "getOrdinal", (ins), [{
      return $_self.getOrdinalAttr().getValue().template cast<IntegerAttr>().getInt();
    }]>,
    InterfaceMethod<[{}], "void", "makeMutable", (ins)>,
    InterfaceMethod<[{}], "void", "clearInitializer", (ins)>,
    InterfaceMethod<[{}], "void", "clearInitialValue", (ins)>,
  ];
}

//===----------------------------------------------------------------------===//
// VM traits
//===----------------------------------------------------------------------===//

// Operations with this trait are pseudo-ops that must be converted to other ops
// prior to serialization. There are no runtime equivalents of the op.
def VM_PseudoOp : NativeOpTrait<"IREE::VM::PseudoOp">;

// Operations with this trait will be stripped in builds where
// debugging-related operations are not needed (such as release builds).
def VM_DebugOnly : NativeOpTrait<"IREE::VM::DebugOnly">;

// Operations with this trait will disallow code motion across them to ensure
// that execution order before and after the barrier op remains the same.
def VM_FullBarrier : NativeOpTrait<"IREE::VM::FullBarrier">;

// Operations with this trait require the VM i64 extension.
def VM_ExtI64 : NativeOpTrait<"IREE::VM::ExtI64">;
// Operations with this trait require the VM f32 extension.
def VM_ExtF32 : NativeOpTrait<"IREE::VM::ExtF32">;
// Operations with this trait require the VM f64 extension.
def VM_ExtF64 : NativeOpTrait<"IREE::VM::ExtF64">;

//===----------------------------------------------------------------------===//
// ref<T> types
//===----------------------------------------------------------------------===//

def VM_AnyRefObject : DialectType<
    IREE_Dialect,
    CPred<"IREE::VM::RefType::isCompatible($_self)">,
    "ref_object"> {
  let typeDescription = [{
    An object that can be referenced by a vm.ref.
  }];
}

def VM_AnyRef : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<IREE::VM::RefType>()">,
    "ref"> {
  let typeDescription = [{
    An iree_vm_ref_t/iree::vm::ref<T> of some type.
  }];
}

class VM_RefOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::VM::RefType>()">,
      SubstLeaves<"$_self", "$_self.cast<IREE::VM::RefType>().getObjectType()",
                  type.predicate>
    ]>, "ref<" # type.description # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::VM::RefType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// List types
//===----------------------------------------------------------------------===//

def VM_AnyList : DialectType<
    IREE_Dialect,
    And<[
      CPred<"$_self.isa<IREE::VM::RefType>()">,
      CPred<"$_self.cast<IREE::VM::RefType>().getObjectType().isa<IREE::VM::ListType>()">,
    ]>,
    "list"> {
  let typeDescription = [{
    An iree_vm_list_t/iree::vm::list<T> of some type.
  }];
}

class VM_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.cast<IREE::VM::RefType>().getObjectType().isa<IREE::VM::ListType>()">,
      SubstLeaves<"$_self",
                  "$_self.cast<IREE::VM::RefType>().getObjectType().cast<IREE::VM::ListType>().getElementType()",
                  type.predicate>
    ]>, "list<" # type.description # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::VM::ListType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// VM types
//===----------------------------------------------------------------------===//

def VM_GlobalRefAttr : AliasedSymbolRefAttr;
def VM_FuncRefAttr : AliasedSymbolRefAttr;

def VM_Ordinal : SignlessIntegerAttrBase<I32, "ordinal value">;

def VM_Ptr : I<32>;

def VM_CondValue : I<32> {
  let typeDescription = [{
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

class VM_ConstIntValueAttr<I type> : Attr<
    Or<[
      SignlessIntegerAttrBase<type,
                              type.bitwidth # "-bit integer value">.predicate,
      SignlessIntElementsAttr<type.bitwidth>.predicate,
    ]>> {
  let storageType = "Attribute";
  let returnType = "Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

#endif  // IREE_DIALECT_VM_BASE
