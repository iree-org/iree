// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

def VM_OPC_ConstF64Zero          : VM_OPC<0x04, "ConstF64Zero">;
def VM_OPC_ConstF64              : VM_OPC<0x05, "ConstF64">;

def VM_OPC_ListGetF64            : VM_OPC<0x06, "ListGetF64">;
def VM_OPC_ListSetF64            : VM_OPC<0x07, "ListSetF64">;

def VM_OPC_SelectF64             : VM_OPC<0x08, "SelectF64">;
def VM_OPC_SwitchF64             : VM_OPC<0x09, "SwitchF64">;

def VM_OPC_AddF64                : VM_OPC<0x0A, "AddF64">;
def VM_OPC_SubF64                : VM_OPC<0x0B, "SubF64">;
def VM_OPC_MulF64                : VM_OPC<0x0C, "MulF64">;
def VM_OPC_DivF64                : VM_OPC<0x0D, "DivF64">;
def VM_OPC_RemF64                : VM_OPC<0x0E, "RemF64">;
def VM_OPC_FMAF64                : VM_OPC<0x0F, "FMAF64">;
def VM_OPC_AbsF64                : VM_OPC<0x10, "AbsF64">;
def VM_OPC_NegF64                : VM_OPC<0x11, "NegF64">;
def VM_OPC_CeilF64               : VM_OPC<0x12, "CeilF64">;
def VM_OPC_FloorF64              : VM_OPC<0x13, "FloorF64">;
def VM_OPC_RoundF64              : VM_OPC<0x3C, "RoundF64">;
def VM_OPC_RoundF64Even          : VM_OPC<0x3F, "RoundF64Even">;
def VM_OPC_TruncF64F32           : VM_OPC<0x14, "TruncF64F32">;
def VM_OPC_ExtF32F64             : VM_OPC<0x15, "ExtF32F64">;
def VM_OPC_MinF64                : VM_OPC<0x3D, "MinF64">;
def VM_OPC_MaxF64                : VM_OPC<0x3E, "MaxF64">;

def VM_OPC_CastSI32F64           : VM_OPC<0x16, "CastSI32F64">;
def VM_OPC_CastUI32F64           : VM_OPC<0x17, "CastUI32F64">;
def VM_OPC_CastF64SI32           : VM_OPC<0x18, "CastF64SI32">;
def VM_OPC_CastF64UI32           : VM_OPC<0x19, "CastF64UI32">;
def VM_OPC_CastSI64F64           : VM_OPC<0x1A, "CastSI64F64">;
def VM_OPC_CastUI64F64           : VM_OPC<0x1B, "CastUI64F64">;
def VM_OPC_CastF64SI64           : VM_OPC<0x1C, "CastF64SI64">;
def VM_OPC_CastF64UI64           : VM_OPC<0x1D, "CastF64UI64">;
def VM_OPC_BitcastI64F64         : VM_OPC<0x1E, "BitcastI64F64">;
def VM_OPC_BitcastF64I64         : VM_OPC<0x1F, "BitcastF64I64">;

def VM_OPC_AtanF64               : VM_OPC<0x20, "AtanF64">;
def VM_OPC_Atan2F64              : VM_OPC<0x21, "Atan2F64">;
def VM_OPC_CosF64                : VM_OPC<0x22, "CosF64">;
def VM_OPC_SinF64                : VM_OPC<0x23, "SinF64">;
def VM_OPC_ExpF64                : VM_OPC<0x24, "ExpF64">;
def VM_OPC_Exp2F64               : VM_OPC<0x25, "Exp2F64">;
def VM_OPC_ExpM1F64              : VM_OPC<0x26, "ExpM1F64">;
def VM_OPC_LogF64                : VM_OPC<0x27, "LogF64">;
def VM_OPC_Log10F64              : VM_OPC<0x28, "Log10F64">;
def VM_OPC_Log1pF64              : VM_OPC<0x29, "Log1pF64">;
def VM_OPC_Log2F64               : VM_OPC<0x2A, "Log2F64">;
def VM_OPC_PowF64                : VM_OPC<0x2B, "PowF64">;
def VM_OPC_RsqrtF64              : VM_OPC<0x2C, "RsqrtF64">;
def VM_OPC_SqrtF64               : VM_OPC<0x2D, "SqrtF64">;
def VM_OPC_TanhF64               : VM_OPC<0x2E, "TanhF64">;
def VM_OPC_ErfF64                : VM_OPC<0x2F, "ErfF64">;

def VM_OPC_CmpEQF64O             : VM_OPC<0x30, "CmpEQF64O">;
def VM_OPC_CmpEQF64U             : VM_OPC<0x31, "CmpEQF64U">;
def VM_OPC_CmpNEF64O             : VM_OPC<0x32, "CmpNEF64O">;
def VM_OPC_CmpNEF64U             : VM_OPC<0x33, "CmpNEF64U">;
def VM_OPC_CmpLTF64O             : VM_OPC<0x34, "CmpLTF64O">;
def VM_OPC_CmpLTF64U             : VM_OPC<0x35, "CmpLTF64U">;
def VM_OPC_CmpLTEF64O            : VM_OPC<0x36, "CmpLTEF64O">;
def VM_OPC_CmpLTEF64U            : VM_OPC<0x37, "CmpLTEF64U">;
def VM_OPC_CmpNaNF64             : VM_OPC<0x38, "CmpNaNF64">;

// Buffer load/store:
def VM_OPC_BufferLoadF64         : VM_OPC<0x39, "BufferLoadF64">;
def VM_OPC_BufferStoreF64        : VM_OPC<0x3A, "BufferStoreF64">;
def VM_OPC_BufferFillF64         : VM_OPC<0x3B, "BufferFillF64">;

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
    VM_OPC_FMAF64,
    VM_OPC_AbsF64,
    VM_OPC_NegF64,
    VM_OPC_CeilF64,
    VM_OPC_FloorF64,
    VM_OPC_RoundF64,
    VM_OPC_MinF64,
    VM_OPC_MaxF64,

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
    VM_OPC_BitcastI64F64,
    VM_OPC_BitcastF64I64,

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
    VM_OPC_ErfF64,

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
