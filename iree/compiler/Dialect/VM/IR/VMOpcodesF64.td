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

#ifndef IREE_DIALECT_VM_OPCODES_F64
#define IREE_DIALECT_VM_OPCODES_F64

include "iree/compiler/Dialect/VM/IR/VMBase.td"
include "iree/compiler/Dialect/VM/IR/VMOpcodesCore.td"

//===----------------------------------------------------------------------===//
// F64 VM Opcode Extension
//===----------------------------------------------------------------------===//
// Ops are encoded as a VM_OPC_ExtF64 + the opcode below.

def VM_OPC_GlobalLoadF64         : VM_OPC<0x00, "GlobalLoadF64">;
def VM_OPC_GlobalStoreF64        : VM_OPC<0x01, "GlobalStoreF64">;
def VM_OPC_GlobalLoadIndirectF64 : VM_OPC<0x02, "GlobalLoadIndirectF64">;
def VM_OPC_GlobalStoreIndirectF64: VM_OPC<0x03, "GlobalStoreIndirectF64">;

def VM_OPC_ConstF64Zero          : VM_OPC<0x08, "ConstF64Zero">;
def VM_OPC_ConstF64              : VM_OPC<0x09, "ConstF64">;

def VM_OPC_ListGetF64            : VM_OPC<0x14, "ListGetF64">;
def VM_OPC_ListSetF64            : VM_OPC<0x15, "ListSetF64">;

def VM_OPC_SelectF64             : VM_OPC<0x1E, "SelectF64">;
def VM_OPC_SwitchF64             : VM_OPC<0x20, "SwitchF64">;

def VM_OPC_AddF64                : VM_OPC<0x22, "AddF64">;
def VM_OPC_SubF64                : VM_OPC<0x23, "SubF64">;
def VM_OPC_MulF64                : VM_OPC<0x24, "MulF64">;
def VM_OPC_DivF64                : VM_OPC<0x25, "DivF64">;
def VM_OPC_RemF64                : VM_OPC<0x26, "RemF64">;
def VM_OPC_AbsF64                : VM_OPC<0x27, "AbsF64">;
def VM_OPC_NegF64                : VM_OPC<0x28, "NegF64">;
def VM_OPC_CeilF64               : VM_OPC<0x29, "CeilF64">;
def VM_OPC_FloorF64              : VM_OPC<0x2A, "FloorF64">;
def VM_OPC_TruncF64F32           : VM_OPC<0x2B, "TruncF64F32">;
def VM_OPC_ExtF32F64             : VM_OPC<0x2C, "ExtF32F64">;

def VM_OPC_CastSI32F64           : VM_OPC<0x30, "CastSI32F64">;
def VM_OPC_CastUI32F64           : VM_OPC<0x31, "CastUI32F64">;
def VM_OPC_CastF64SI32           : VM_OPC<0x32, "CastF64SI32">;
def VM_OPC_CastF64UI32           : VM_OPC<0x33, "CastF64UI32">;
def VM_OPC_CastSI64F64           : VM_OPC<0x34, "CastSI64F64">;
def VM_OPC_CastUI64F64           : VM_OPC<0x35, "CastUI64F64">;
def VM_OPC_CastF64SI64           : VM_OPC<0x36, "CastF64SI64">;
def VM_OPC_CastF64UI64           : VM_OPC<0x37, "CastF64UI64">;

def VM_OPC_AtanF64               : VM_OPC<0x40, "AtanF64">;
def VM_OPC_Atan2F64              : VM_OPC<0x41, "Atan2F64">;
def VM_OPC_CosF64                : VM_OPC<0x42, "CosF64">;
def VM_OPC_SinF64                : VM_OPC<0x43, "SinF64">;
def VM_OPC_ExpF64                : VM_OPC<0x44, "ExpF64">;
def VM_OPC_Exp2F64               : VM_OPC<0x45, "Exp2F64">;
def VM_OPC_ExpM1F64              : VM_OPC<0x46, "ExpM1F64">;
def VM_OPC_LogF64                : VM_OPC<0x47, "LogF64">;
def VM_OPC_Log10F64              : VM_OPC<0x48, "Log10F64">;
def VM_OPC_Log1pF64              : VM_OPC<0x49, "Log1pF64">;
def VM_OPC_Log2F64               : VM_OPC<0x4A, "Log2F64">;
def VM_OPC_PowF64                : VM_OPC<0x4B, "PowF64">;
def VM_OPC_RsqrtF64              : VM_OPC<0x4C, "RsqrtF64">;
def VM_OPC_SqrtF64               : VM_OPC<0x4D, "SqrtF64">;
def VM_OPC_TanhF64               : VM_OPC<0x4E, "TanhF64">;

def VM_OPC_CmpEQF64O             : VM_OPC<0x60, "CmpEQF64O">;
def VM_OPC_CmpEQF64U             : VM_OPC<0x61, "CmpEQF64U">;
def VM_OPC_CmpNEF64O             : VM_OPC<0x62, "CmpNEF64O">;
def VM_OPC_CmpNEF64U             : VM_OPC<0x63, "CmpNEF64U">;
def VM_OPC_CmpLTF64O             : VM_OPC<0x64, "CmpLTF64O">;
def VM_OPC_CmpLTF64U             : VM_OPC<0x65, "CmpLTF64U">;
def VM_OPC_CmpLTEF64O            : VM_OPC<0x66, "CmpLTEF64O">;
def VM_OPC_CmpLTEF64U            : VM_OPC<0x67, "CmpLTEF64U">;
def VM_OPC_CmpNaNF64             : VM_OPC<0x70, "CmpNaNF64">;

// Buffer load/store:
def VM_OPC_BufferLoadF64         : VM_OPC<0xB0, "BufferLoadF64">;
def VM_OPC_BufferStoreF64        : VM_OPC<0xB1, "BufferStoreF64">;
def VM_OPC_BufferFillF64         : VM_OPC<0xC0, "BufferFillF64">;

// Runtime enum iree_vm_ext_f64_op_t:
def VM_ExtF64OpcodeAttr :
    VM_OPC_EnumAttr<"ExtF64Opcode",
                    "iree_vm_ext_f64_op_t",
                    "EXT_F64",  // IREE_VM_OP_EXT_F64_*
                    "valid VM operation encodings in the f64 extension",
                    VM_OPC_PrefixExtF64, [
    VM_OPC_GlobalLoadF64,
    VM_OPC_GlobalStoreF64,
    VM_OPC_GlobalLoadIndirectF64,
    VM_OPC_GlobalStoreIndirectF64,

    VM_OPC_ConstF64Zero,
    VM_OPC_ConstF64,

    VM_OPC_ListGetF64,
    VM_OPC_ListSetF64,

    VM_OPC_SelectF64,
    VM_OPC_SwitchF64,

    VM_OPC_AddF64,
    VM_OPC_SubF64,
    VM_OPC_MulF64,
    VM_OPC_DivF64,
    VM_OPC_RemF64,
    VM_OPC_AbsF64,
    VM_OPC_NegF64,
    VM_OPC_CeilF64,
    VM_OPC_FloorF64,

    VM_OPC_TruncF64F32,
    VM_OPC_ExtF32F64,
    VM_OPC_CastSI32F64,
    VM_OPC_CastUI32F64,
    VM_OPC_CastF64SI32,
    VM_OPC_CastF64UI32,
    VM_OPC_CastSI64F64,
    VM_OPC_CastUI64F64,
    VM_OPC_CastF64SI64,
    VM_OPC_CastF64UI64,

    VM_OPC_AtanF64,
    VM_OPC_Atan2F64,
    VM_OPC_CosF64,
    VM_OPC_SinF64,
    VM_OPC_ExpF64,
    VM_OPC_Exp2F64,
    VM_OPC_ExpM1F64,
    VM_OPC_LogF64,
    VM_OPC_Log10F64,
    VM_OPC_Log1pF64,
    VM_OPC_Log2F64,
    VM_OPC_PowF64,
    VM_OPC_RsqrtF64,
    VM_OPC_SqrtF64,
    VM_OPC_TanhF64,

    VM_OPC_CmpEQF64O,
    VM_OPC_CmpEQF64U,
    VM_OPC_CmpNEF64O,
    VM_OPC_CmpNEF64U,
    VM_OPC_CmpLTF64O,
    VM_OPC_CmpLTF64U,
    VM_OPC_CmpLTEF64O,
    VM_OPC_CmpLTEF64U,
    VM_OPC_CmpNaNF64,

    VM_OPC_BufferLoadF64,
    VM_OPC_BufferStoreF64,
    VM_OPC_BufferFillF64,
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_F64
