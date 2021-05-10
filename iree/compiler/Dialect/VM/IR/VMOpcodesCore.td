// Copyright 2021 Google LLC
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

#ifndef IREE_DIALECT_VM_OPCODES_CORE
#define IREE_DIALECT_VM_OPCODES_CORE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

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

// Integer arithmetic:
def VM_OPC_AddI32                : VM_OPC<0x22, "AddI32">;
def VM_OPC_SubI32                : VM_OPC<0x23, "SubI32">;
def VM_OPC_MulI32                : VM_OPC<0x24, "MulI32">;
def VM_OPC_DivI32S               : VM_OPC<0x25, "DivI32S">;
def VM_OPC_DivI32U               : VM_OPC<0x26, "DivI32U">;
def VM_OPC_RemI32S               : VM_OPC<0x27, "RemI32S">;
def VM_OPC_RemI32U               : VM_OPC<0x28, "RemI32U">;

// Integer bit manipulation:
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

// Buffer load/store:
// NOTE: though not used today the opcodes are chosen to allow for bit magic to
// reduce dispatch overhead:
//                bit 3             bit 2          bit 1-0
//        +------------+-----------------+----------------+
//  0xB...| load/store | unsigned/signed | byte count - 1 |
//        +------------+-----------------+----------------+
def VM_OPC_BufferLoadI8U         : VM_OPC<0xB0, "BufferLoadI8U">;
def VM_OPC_BufferLoadI8S         : VM_OPC<0xB4, "BufferLoadI8S">;
def VM_OPC_BufferLoadI16U        : VM_OPC<0xB1, "BufferLoadI16U">;
def VM_OPC_BufferLoadI16S        : VM_OPC<0xB5, "BufferLoadI16S">;
def VM_OPC_BufferLoadI32         : VM_OPC<0xB3, "BufferLoadI32">;
def VM_OPC_BufferStoreI8         : VM_OPC<0xB8, "BufferStoreI8">;
def VM_OPC_BufferStoreI16        : VM_OPC<0xB9, "BufferStoreI16">;
def VM_OPC_BufferStoreI32        : VM_OPC<0xBB, "BufferStoreI32">;

// Buffer manipulation:
def VM_OPC_BufferAlloc           : VM_OPC<0xC0, "BufferAlloc">;
def VM_OPC_BufferClone           : VM_OPC<0xC1, "BufferClone">;
def VM_OPC_BufferLength          : VM_OPC<0xC2, "BufferLength">;
def VM_OPC_BufferCopy            : VM_OPC<0xC6, "BufferCopy">;
def VM_OPC_BufferCompare         : VM_OPC<0xC7, "BufferCompare">;
// NOTE: lower 2 bits are byte count - 1.
//          b3  b2          bit 1-0
//        +---+---+----------------+
//  0xC...| 1 | 1 | byte count - 1 |
//        +---+---+----------------+
def VM_OPC_BufferFillI8          : VM_OPC<0xCC, "BufferFillI8">;
def VM_OPC_BufferFillI16         : VM_OPC<0xCD, "BufferFillI16">;
def VM_OPC_BufferFillI32         : VM_OPC<0xCF, "BufferFillI32">;

// Extension prefixes:
def VM_OPC_PrefixExtI64          : VM_OPC<0xE0, "PrefixExtI64">;
def VM_OPC_PrefixExtF32          : VM_OPC<0xE1, "PrefixExtF32">;
def VM_OPC_PrefixExtF64          : VM_OPC<0xE2, "PrefixExtF64">;

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

    VM_OPC_BufferLoadI8U,
    VM_OPC_BufferLoadI8S,
    VM_OPC_BufferLoadI16U,
    VM_OPC_BufferLoadI16S,
    VM_OPC_BufferLoadI32,
    VM_OPC_BufferStoreI8,
    VM_OPC_BufferStoreI16,
    VM_OPC_BufferStoreI32,

    VM_OPC_BufferAlloc,
    VM_OPC_BufferClone,
    VM_OPC_BufferLength,
    VM_OPC_BufferFillI8,
    VM_OPC_BufferFillI16,
    VM_OPC_BufferFillI32,
    VM_OPC_BufferCopy,
    VM_OPC_BufferCompare,

    // Extension opcodes (0xE0-0xFF):
    VM_OPC_PrefixExtI64,  // VM_ExtI64OpcodeAttr
    VM_OPC_PrefixExtF32,  // VM_ExtF32OpcodeAttr
    VM_OPC_PrefixExtF64,  // VM_ExtF64OpcodeAttr
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_CORE
