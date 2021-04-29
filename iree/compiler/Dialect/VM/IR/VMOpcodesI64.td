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

#ifndef IREE_DIALECT_VM_OPCODES_I64
#define IREE_DIALECT_VM_OPCODES_I64

include "iree/compiler/Dialect/VM/IR/VMBase.td"
include "iree/compiler/Dialect/VM/IR/VMOpcodesCore.td"

//===----------------------------------------------------------------------===//
// I64 VM Opcode Extension
//===----------------------------------------------------------------------===//
// Ops are encoded as a VM_OPC_ExtI64 + the opcode below.

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

// Buffer load/store:
def VM_OPC_BufferLoadI64         : VM_OPC<0xB0, "BufferLoadI64">;
def VM_OPC_BufferStoreI64        : VM_OPC<0xB1, "BufferStoreI64">;
def VM_OPC_BufferFillI64         : VM_OPC<0xC0, "BufferFillI64">;

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

    VM_OPC_BufferLoadI64,
    VM_OPC_BufferStoreI64,
    VM_OPC_BufferFillI64,
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_I64
