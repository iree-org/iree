// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VM_OPCODES_CORE
#define IREE_DIALECT_VM_OPCODES_CORE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

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
// is optional. An op with the ExtF64 trait will require VM_OPC_ExtF64, for
// example. Ops that bridge extension sets have a canonical form that may
// require multiple prefix codes (for example, the i64<->f64 extensions).

class VM_OPC<int opcode, string name> :
    IntEnumAttrCaseBase<I8, name, name, opcode>;

class VM_OPC_EnumAttr<string name, string enumName, string enumTag,
                      string description,
                      VM_OPC prefix = ?,
                      list<VM_OPC> cases> :
    IntEnumAttr<I8, name, description, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::VM";
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

// Next available opcode: 0x83

// Globals:
def VM_OPC_GlobalLoadI32         : VM_OPC<0x00, "GlobalLoadI32">;
def VM_OPC_GlobalStoreI32        : VM_OPC<0x01, "GlobalStoreI32">;
def VM_OPC_GlobalLoadIndirectI32 : VM_OPC<0x02, "GlobalLoadIndirectI32">;
def VM_OPC_GlobalStoreIndirectI32: VM_OPC<0x03, "GlobalStoreIndirectI32">;
def VM_OPC_GlobalLoadI64         : VM_OPC<0x04, "GlobalLoadI64">;
def VM_OPC_GlobalStoreI64        : VM_OPC<0x05, "GlobalStoreI64">;
def VM_OPC_GlobalLoadIndirectI64 : VM_OPC<0x06, "GlobalLoadIndirectI64">;
def VM_OPC_GlobalStoreIndirectI64: VM_OPC<0x07, "GlobalStoreIndirectI64">;
def VM_OPC_GlobalLoadRef         : VM_OPC<0x08, "GlobalLoadRef">;
def VM_OPC_GlobalStoreRef        : VM_OPC<0x09, "GlobalStoreRef">;
def VM_OPC_GlobalLoadIndirectRef : VM_OPC<0x0A, "GlobalLoadIndirectRef">;
def VM_OPC_GlobalStoreIndirectRef: VM_OPC<0x0B, "GlobalStoreIndirectRef">;

// Constants:
def VM_OPC_ConstI32Zero          : VM_OPC<0x0C, "ConstI32Zero">;
def VM_OPC_ConstI32              : VM_OPC<0x0D, "ConstI32">;
def VM_OPC_ConstI64Zero          : VM_OPC<0x0E, "ConstI64Zero">;
def VM_OPC_ConstI64              : VM_OPC<0x0F, "ConstI64">;
def VM_OPC_ConstRefZero          : VM_OPC<0x10, "ConstRefZero">;
def VM_OPC_ConstRefRodata        : VM_OPC<0x11, "ConstRefRodata">;

// List operations:
def VM_OPC_ListAlloc             : VM_OPC<0x12, "ListAlloc">;
def VM_OPC_ListReserve           : VM_OPC<0x13, "ListReserve">;
def VM_OPC_ListSize              : VM_OPC<0x14, "ListSize">;
def VM_OPC_ListResize            : VM_OPC<0x15, "ListResize">;
def VM_OPC_ListGetI32            : VM_OPC<0x16, "ListGetI32">;
def VM_OPC_ListSetI32            : VM_OPC<0x17, "ListSetI32">;
def VM_OPC_ListGetI64            : VM_OPC<0x18, "ListGetI64">;
def VM_OPC_ListSetI64            : VM_OPC<0x19, "ListSetI64">;
def VM_OPC_ListGetRef            : VM_OPC<0x1A, "ListGetRef">;
def VM_OPC_ListSetRef            : VM_OPC<0x1B, "ListSetRef">;
// RESERVED: push.i32
// RESERVED: pop.i32
// RESERVED: copy to other list
// RESERVED: slice clone into new list
// RESERVED: read byte buffer?
// RESERVED: write byte buffer?

// Conditional assignment:
def VM_OPC_SelectI32             : VM_OPC<0x1C, "SelectI32">;
def VM_OPC_SelectI64             : VM_OPC<0x1D, "SelectI64">;
def VM_OPC_SelectRef             : VM_OPC<0x1E, "SelectRef">;
def VM_OPC_SwitchI32             : VM_OPC<0x1F, "SwitchI32">;
def VM_OPC_SwitchI64             : VM_OPC<0x20, "SwitchI64">;
def VM_OPC_SwitchRef             : VM_OPC<0x21, "SwitchRef">;

// 32-bit integer arithmetic:
def VM_OPC_AddI32                : VM_OPC<0x22, "AddI32">;
def VM_OPC_SubI32                : VM_OPC<0x23, "SubI32">;
def VM_OPC_MulI32                : VM_OPC<0x24, "MulI32">;
def VM_OPC_DivI32S               : VM_OPC<0x25, "DivI32S">;
def VM_OPC_DivI32U               : VM_OPC<0x26, "DivI32U">;
def VM_OPC_RemI32S               : VM_OPC<0x27, "RemI32S">;
def VM_OPC_RemI32U               : VM_OPC<0x28, "RemI32U">;
def VM_OPC_FMAI32                : VM_OPC<0x29, "FMAI32">;
def VM_OPC_AbsI32                : VM_OPC<0x77, "AbsI32">;
def VM_OPC_MinI32S               : VM_OPC<0x7A, "MinI32S">;
def VM_OPC_MinI32U               : VM_OPC<0x7B, "MinI32U">;
def VM_OPC_MaxI32S               : VM_OPC<0x7C, "MaxI32S">;
def VM_OPC_MaxI32U               : VM_OPC<0x7D, "MaxI32U">;

// 64-bit integer arithmetic:
def VM_OPC_AddI64                : VM_OPC<0x2A, "AddI64">;
def VM_OPC_SubI64                : VM_OPC<0x2B, "SubI64">;
def VM_OPC_MulI64                : VM_OPC<0x2C, "MulI64">;
def VM_OPC_DivI64S               : VM_OPC<0x2D, "DivI64S">;
def VM_OPC_DivI64U               : VM_OPC<0x2E, "DivI64U">;
def VM_OPC_RemI64S               : VM_OPC<0x2F, "RemI64S">;
def VM_OPC_RemI64U               : VM_OPC<0x30, "RemI64U">;
def VM_OPC_FMAI64                : VM_OPC<0x31, "FMAI64">;
def VM_OPC_AbsI64                : VM_OPC<0x78, "AbsI64">;
def VM_OPC_MinI64S               : VM_OPC<0x7E, "MinI64S">;
def VM_OPC_MinI64U               : VM_OPC<0x7F, "MinI64U">;
def VM_OPC_MaxI64S               : VM_OPC<0x80, "MaxI64S">;
def VM_OPC_MaxI64U               : VM_OPC<0x81, "MaxI64U">;

// 32-bit integer bit manipulation:
def VM_OPC_NotI32                : VM_OPC<0x32, "NotI32">;
def VM_OPC_AndI32                : VM_OPC<0x33, "AndI32">;
def VM_OPC_OrI32                 : VM_OPC<0x34, "OrI32">;
def VM_OPC_XorI32                : VM_OPC<0x35, "XorI32">;
def VM_OPC_CtlzI32               : VM_OPC<0x75, "CtlzI32">;

// 64-bit integer bit manipulation:
def VM_OPC_NotI64                : VM_OPC<0x36, "NotI64">;
def VM_OPC_AndI64                : VM_OPC<0x37, "AndI64">;
def VM_OPC_OrI64                 : VM_OPC<0x38, "OrI64">;
def VM_OPC_XorI64                : VM_OPC<0x39, "XorI64">;
def VM_OPC_CtlzI64               : VM_OPC<0x76, "CtlzI64">;

// 32-bit bitwise shifts and rotates:
def VM_OPC_ShlI32                : VM_OPC<0x3A, "ShlI32">;
def VM_OPC_ShrI32S               : VM_OPC<0x3B, "ShrI32S">;
def VM_OPC_ShrI32U               : VM_OPC<0x3C, "ShrI32U">;

// 64-bit bitwise shifts and rotates:
def VM_OPC_ShlI64                : VM_OPC<0x3D, "ShlI64">;
def VM_OPC_ShrI64S               : VM_OPC<0x3E, "ShrI64S">;
def VM_OPC_ShrI64U               : VM_OPC<0x3F, "ShrI64U">;

// Casting and type conversion/emulation:
def VM_OPC_TruncI32I8            : VM_OPC<0x40, "TruncI32I8">;
def VM_OPC_TruncI32I16           : VM_OPC<0x41, "TruncI32I16">;
def VM_OPC_TruncI64I32           : VM_OPC<0x42, "TruncI64I32">;
def VM_OPC_ExtI8I32S             : VM_OPC<0x43, "ExtI8I32S">;
def VM_OPC_ExtI8I32U             : VM_OPC<0x44, "ExtI8I32U">;
def VM_OPC_ExtI16I32S            : VM_OPC<0x45, "ExtI16I32S">;
def VM_OPC_ExtI16I32U            : VM_OPC<0x46, "ExtI16I32U">;
def VM_OPC_ExtI32I64S            : VM_OPC<0x47, "ExtI32I64S">;
def VM_OPC_ExtI32I64U            : VM_OPC<0x48, "ExtI32I64U">;
def VM_OPC_CastAnyRef            : VM_OPC<0x82, "CastAnyRef">;

// Reduction arithmetic:

// Comparison ops:
def VM_OPC_CmpEQI32              : VM_OPC<0x49, "CmpEQI32">;
def VM_OPC_CmpNEI32              : VM_OPC<0x4A, "CmpNEI32">;
def VM_OPC_CmpLTI32S             : VM_OPC<0x4B, "CmpLTI32S">;
def VM_OPC_CmpLTI32U             : VM_OPC<0x4C, "CmpLTI32U">;
def VM_OPC_CmpNZI32              : VM_OPC<0x4D, "CmpNZI32">;
def VM_OPC_CmpEQI64              : VM_OPC<0x4E, "CmpEQI64">;
def VM_OPC_CmpNEI64              : VM_OPC<0x4F, "CmpNEI64">;
def VM_OPC_CmpLTI64S             : VM_OPC<0x50, "CmpLTI64S">;
def VM_OPC_CmpLTI64U             : VM_OPC<0x51, "CmpLTI64U">;
def VM_OPC_CmpNZI64              : VM_OPC<0x52, "CmpNZI64">;
def VM_OPC_CmpEQRef              : VM_OPC<0x53, "CmpEQRef">;
def VM_OPC_CmpNERef              : VM_OPC<0x54, "CmpNERef">;
def VM_OPC_CmpNZRef              : VM_OPC<0x55, "CmpNZRef">;

// Control flow:
def VM_OPC_Block                 : VM_OPC<0x79, "Block">;
def VM_OPC_Branch                : VM_OPC<0x56, "Branch">;
def VM_OPC_CondBranch            : VM_OPC<0x57, "CondBranch">;
def VM_OPC_Call                  : VM_OPC<0x58, "Call">;
def VM_OPC_CallVariadic          : VM_OPC<0x59, "CallVariadic">;
def VM_OPC_Return                : VM_OPC<0x5A, "Return">;
def VM_OPC_Fail                  : VM_OPC<0x5B, "Fail">;
def VM_OPC_ImportResolved        : VM_OPC<0x5C, "ImportResolved">;

// Async/fiber ops:
def VM_OPC_Yield                 : VM_OPC<0x5D, "Yield">;

// Debugging:
def VM_OPC_Trace                 : VM_OPC<0x5E, "Trace">;
def VM_OPC_Print                 : VM_OPC<0x5F, "Print">;
def VM_OPC_CondBreak             : VM_OPC<0x60, "CondBreak">;
def VM_OPC_Break                 : VM_OPC<0x61, "Break">;

// Buffer load/store:
def VM_OPC_BufferLoadI8U         : VM_OPC<0x62, "BufferLoadI8U">;
def VM_OPC_BufferLoadI8S         : VM_OPC<0x63, "BufferLoadI8S">;
def VM_OPC_BufferLoadI16U        : VM_OPC<0x64, "BufferLoadI16U">;
def VM_OPC_BufferLoadI16S        : VM_OPC<0x65, "BufferLoadI16S">;
def VM_OPC_BufferLoadI32         : VM_OPC<0x66, "BufferLoadI32">;
def VM_OPC_BufferLoadI64         : VM_OPC<0x67, "BufferLoadI64">;
def VM_OPC_BufferStoreI8         : VM_OPC<0x68, "BufferStoreI8">;
def VM_OPC_BufferStoreI16        : VM_OPC<0x69, "BufferStoreI16">;
def VM_OPC_BufferStoreI32        : VM_OPC<0x6A, "BufferStoreI32">;
def VM_OPC_BufferStoreI64        : VM_OPC<0x6B, "BufferStoreI64">;

// Buffer manipulation:
def VM_OPC_BufferAlloc           : VM_OPC<0x6C, "BufferAlloc">;
def VM_OPC_BufferClone           : VM_OPC<0x6D, "BufferClone">;
def VM_OPC_BufferLength          : VM_OPC<0x6E, "BufferLength">;
def VM_OPC_BufferCopy            : VM_OPC<0x6F, "BufferCopy">;
def VM_OPC_BufferCompare         : VM_OPC<0x70, "BufferCompare">;
def VM_OPC_BufferFillI8          : VM_OPC<0x71, "BufferFillI8">;
def VM_OPC_BufferFillI16         : VM_OPC<0x72, "BufferFillI16">;
def VM_OPC_BufferFillI32         : VM_OPC<0x73, "BufferFillI32">;
def VM_OPC_BufferFillI64         : VM_OPC<0x74, "BufferFillI64">;

// Extension prefixes:
def VM_OPC_PrefixExtF32          : VM_OPC<0xE0, "PrefixExtF32">;
def VM_OPC_PrefixExtF64          : VM_OPC<0xE1, "PrefixExtF64">;

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
    VM_OPC_GlobalLoadI64,
    VM_OPC_GlobalStoreI64,
    VM_OPC_GlobalLoadIndirectI64,
    VM_OPC_GlobalStoreIndirectI64,
    VM_OPC_GlobalLoadRef,
    VM_OPC_GlobalStoreRef,
    VM_OPC_GlobalLoadIndirectRef,
    VM_OPC_GlobalStoreIndirectRef,

    VM_OPC_ConstI32Zero,
    VM_OPC_ConstI32,
    VM_OPC_ConstI64Zero,
    VM_OPC_ConstI64,
    VM_OPC_ConstRefZero,
    VM_OPC_ConstRefRodata,

    VM_OPC_ListAlloc,
    VM_OPC_ListReserve,
    VM_OPC_ListSize,
    VM_OPC_ListResize,
    VM_OPC_ListGetI32,
    VM_OPC_ListSetI32,
    VM_OPC_ListGetI64,
    VM_OPC_ListSetI64,
    VM_OPC_ListGetRef,
    VM_OPC_ListSetRef,

    VM_OPC_SelectI32,
    VM_OPC_SelectI64,
    VM_OPC_SelectRef,
    VM_OPC_SwitchI32,
    VM_OPC_SwitchI64,
    VM_OPC_SwitchRef,

    VM_OPC_AddI32,
    VM_OPC_SubI32,
    VM_OPC_MulI32,
    VM_OPC_DivI32S,
    VM_OPC_DivI32U,
    VM_OPC_RemI32S,
    VM_OPC_RemI32U,
    VM_OPC_FMAI32,
    VM_OPC_AbsI32,
    VM_OPC_MinI32S,
    VM_OPC_MinI32U,
    VM_OPC_MaxI32S,
    VM_OPC_MaxI32U,

    VM_OPC_AddI64,
    VM_OPC_SubI64,
    VM_OPC_MulI64,
    VM_OPC_DivI64S,
    VM_OPC_DivI64U,
    VM_OPC_RemI64S,
    VM_OPC_RemI64U,
    VM_OPC_FMAI64,
    VM_OPC_AbsI64,
    VM_OPC_MinI64S,
    VM_OPC_MinI64U,
    VM_OPC_MaxI64S,
    VM_OPC_MaxI64U,

    VM_OPC_NotI32,
    VM_OPC_AndI32,
    VM_OPC_OrI32,
    VM_OPC_XorI32,
    VM_OPC_CtlzI32,

    VM_OPC_NotI64,
    VM_OPC_AndI64,
    VM_OPC_OrI64,
    VM_OPC_XorI64,
    VM_OPC_CtlzI64,

    VM_OPC_ShlI32,
    VM_OPC_ShrI32S,
    VM_OPC_ShrI32U,

    VM_OPC_ShlI64,
    VM_OPC_ShrI64S,
    VM_OPC_ShrI64U,

    VM_OPC_TruncI32I8,
    VM_OPC_TruncI32I16,
    VM_OPC_TruncI64I32,
    VM_OPC_ExtI8I32S,
    VM_OPC_ExtI8I32U,
    VM_OPC_ExtI16I32S,
    VM_OPC_ExtI16I32U,
    VM_OPC_ExtI32I64S,
    VM_OPC_ExtI32I64U,
    VM_OPC_CastAnyRef,

    VM_OPC_CmpEQI32,
    VM_OPC_CmpNEI32,
    VM_OPC_CmpLTI32S,
    VM_OPC_CmpLTI32U,
    VM_OPC_CmpNZI32,
    VM_OPC_CmpEQI64,
    VM_OPC_CmpNEI64,
    VM_OPC_CmpLTI64S,
    VM_OPC_CmpLTI64U,
    VM_OPC_CmpNZI64,
    VM_OPC_CmpEQRef,
    VM_OPC_CmpNERef,
    VM_OPC_CmpNZRef,

    VM_OPC_Branch,
    VM_OPC_CondBranch,
    VM_OPC_Call,
    VM_OPC_CallVariadic,
    VM_OPC_Return,
    VM_OPC_Fail,
    VM_OPC_ImportResolved,
    VM_OPC_Yield,
    VM_OPC_Trace,
    VM_OPC_Print,
    VM_OPC_CondBreak,
    VM_OPC_Break,

    VM_OPC_BufferLoadI8U,
    VM_OPC_BufferLoadI8S,
    VM_OPC_BufferLoadI16U,
    VM_OPC_BufferLoadI16S,
    VM_OPC_BufferLoadI32,
    VM_OPC_BufferLoadI64,
    VM_OPC_BufferStoreI8,
    VM_OPC_BufferStoreI16,
    VM_OPC_BufferStoreI32,
    VM_OPC_BufferStoreI64,

    VM_OPC_BufferAlloc,
    VM_OPC_BufferClone,
    VM_OPC_BufferLength,
    VM_OPC_BufferFillI8,
    VM_OPC_BufferFillI16,
    VM_OPC_BufferFillI32,
    VM_OPC_BufferFillI64,
    VM_OPC_BufferCopy,
    VM_OPC_BufferCompare,

    VM_OPC_Block,

    // Extension opcodes (0xE0-0xFF):
    VM_OPC_PrefixExtF32,  // VM_ExtF32OpcodeAttr
    VM_OPC_PrefixExtF64,  // VM_ExtF64OpcodeAttr
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_CORE
