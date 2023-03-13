// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VM_OPCODES_F32
#define IREE_DIALECT_VM_OPCODES_F32

include "iree/compiler/Dialect/VM/IR/VMBase.td"
include "iree/compiler/Dialect/VM/IR/VMOpcodesCore.td"

//===----------------------------------------------------------------------===//
// F32 VM Opcode Extension
//===----------------------------------------------------------------------===//
// Ops are encoded as a VM_OPC_ExtF32 + the opcode below.

def VM_OPC_GlobalLoadF32         : VM_OPC<0x00, "GlobalLoadF32">;
def VM_OPC_GlobalStoreF32        : VM_OPC<0x01, "GlobalStoreF32">;
def VM_OPC_GlobalLoadIndirectF32 : VM_OPC<0x02, "GlobalLoadIndirectF32">;
def VM_OPC_GlobalStoreIndirectF32: VM_OPC<0x03, "GlobalStoreIndirectF32">;

def VM_OPC_ConstF32Zero          : VM_OPC<0x04, "ConstF32Zero">;
def VM_OPC_ConstF32              : VM_OPC<0x05, "ConstF32">;

def VM_OPC_ListGetF32            : VM_OPC<0x06, "ListGetF32">;
def VM_OPC_ListSetF32            : VM_OPC<0x07, "ListSetF32">;

def VM_OPC_SelectF32             : VM_OPC<0x08, "SelectF32">;
def VM_OPC_SwitchF32             : VM_OPC<0x09, "SwitchF32">;

def VM_OPC_AddF32                : VM_OPC<0x0A, "AddF32">;
def VM_OPC_SubF32                : VM_OPC<0x0B, "SubF32">;
def VM_OPC_MulF32                : VM_OPC<0x0C, "MulF32">;
def VM_OPC_DivF32                : VM_OPC<0x0D, "DivF32">;
def VM_OPC_RemF32                : VM_OPC<0x0E, "RemF32">;
def VM_OPC_FMAF32                : VM_OPC<0x0F, "FMAF32">;
def VM_OPC_AbsF32                : VM_OPC<0x10, "AbsF32">;
def VM_OPC_NegF32                : VM_OPC<0x11, "NegF32">;
def VM_OPC_CeilF32               : VM_OPC<0x12, "CeilF32">;
def VM_OPC_FloorF32              : VM_OPC<0x13, "FloorF32">;
def VM_OPC_RoundF32              : VM_OPC<0x36, "RoundF32">;
def VM_OPC_MinF32                : VM_OPC<0x37, "MinF32">;
def VM_OPC_MaxF32                : VM_OPC<0x38, "MaxF32">;

def VM_OPC_CastSI32F32           : VM_OPC<0x14, "CastSI32F32">;
def VM_OPC_CastUI32F32           : VM_OPC<0x15, "CastUI32F32">;
def VM_OPC_CastF32SI32           : VM_OPC<0x16, "CastF32SI32">;
def VM_OPC_CastF32UI32           : VM_OPC<0x17, "CastF32UI32">;
def VM_OPC_BitcastI32F32         : VM_OPC<0x18, "BitcastI32F32">;
def VM_OPC_BitcastF32I32         : VM_OPC<0x19, "BitcastF32I32">;

def VM_OPC_AtanF32               : VM_OPC<0x1A, "AtanF32">;
def VM_OPC_Atan2F32              : VM_OPC<0x1B, "Atan2F32">;
def VM_OPC_CosF32                : VM_OPC<0x1C, "CosF32">;
def VM_OPC_SinF32                : VM_OPC<0x1D, "SinF32">;
def VM_OPC_ExpF32                : VM_OPC<0x1E, "ExpF32">;
def VM_OPC_Exp2F32               : VM_OPC<0x1F, "Exp2F32">;
def VM_OPC_ExpM1F32              : VM_OPC<0x20, "ExpM1F32">;
def VM_OPC_LogF32                : VM_OPC<0x21, "LogF32">;
def VM_OPC_Log10F32              : VM_OPC<0x22, "Log10F32">;
def VM_OPC_Log1pF32              : VM_OPC<0x23, "Log1pF32">;
def VM_OPC_Log2F32               : VM_OPC<0x24, "Log2F32">;
def VM_OPC_PowF32                : VM_OPC<0x25, "PowF32">;
def VM_OPC_RsqrtF32              : VM_OPC<0x26, "RsqrtF32">;
def VM_OPC_SqrtF32               : VM_OPC<0x27, "SqrtF32">;
def VM_OPC_TanhF32               : VM_OPC<0x28, "TanhF32">;
def VM_OPC_ErfF32                : VM_OPC<0x29, "ErfF32">;

def VM_OPC_CmpEQF32O             : VM_OPC<0x2A, "CmpEQF32O">;
def VM_OPC_CmpEQF32U             : VM_OPC<0x2B, "CmpEQF32U">;
def VM_OPC_CmpNEF32O             : VM_OPC<0x2C, "CmpNEF32O">;
def VM_OPC_CmpNEF32U             : VM_OPC<0x2D, "CmpNEF32U">;
def VM_OPC_CmpLTF32O             : VM_OPC<0x2E, "CmpLTF32O">;
def VM_OPC_CmpLTF32U             : VM_OPC<0x2F, "CmpLTF32U">;
def VM_OPC_CmpLTEF32O            : VM_OPC<0x30, "CmpLTEF32O">;
def VM_OPC_CmpLTEF32U            : VM_OPC<0x31, "CmpLTEF32U">;
def VM_OPC_CmpNaNF32             : VM_OPC<0x32, "CmpNaNF32">;

def VM_OPC_BufferLoadF32         : VM_OPC<0x33, "BufferLoadF32">;
def VM_OPC_BufferStoreF32        : VM_OPC<0x34, "BufferStoreF32">;
def VM_OPC_BufferFillF32         : VM_OPC<0x35, "BufferFillF32">;

// Runtime enum iree_vm_ext_f32_op_t:
def VM_ExtF32OpcodeAttr :
    VM_OPC_EnumAttr<"ExtF32Opcode",
                    "iree_vm_ext_f32_op_t",
                    "EXT_F32",  // IREE_VM_OP_EXT_F32_*
                    "valid VM operation encodings in the f32 extension",
                    VM_OPC_PrefixExtF32, [
    VM_OPC_GlobalLoadF32,
    VM_OPC_GlobalStoreF32,
    VM_OPC_GlobalLoadIndirectF32,
    VM_OPC_GlobalStoreIndirectF32,

    VM_OPC_ConstF32Zero,
    VM_OPC_ConstF32,

    VM_OPC_ListGetF32,
    VM_OPC_ListSetF32,

    VM_OPC_SelectF32,
    VM_OPC_SwitchF32,

    VM_OPC_AddF32,
    VM_OPC_SubF32,
    VM_OPC_MulF32,
    VM_OPC_DivF32,
    VM_OPC_RemF32,
    VM_OPC_FMAF32,
    VM_OPC_AbsF32,
    VM_OPC_NegF32,
    VM_OPC_CeilF32,
    VM_OPC_FloorF32,
    VM_OPC_RoundF32,
    VM_OPC_MinF32,
    VM_OPC_MaxF32,

    VM_OPC_CastSI32F32,
    VM_OPC_CastUI32F32,
    VM_OPC_CastF32SI32,
    VM_OPC_CastF32UI32,
    VM_OPC_BitcastI32F32,
    VM_OPC_BitcastF32I32,

    VM_OPC_AtanF32,
    VM_OPC_Atan2F32,
    VM_OPC_CosF32,
    VM_OPC_SinF32,
    VM_OPC_ExpF32,
    VM_OPC_Exp2F32,
    VM_OPC_ExpM1F32,
    VM_OPC_LogF32,
    VM_OPC_Log10F32,
    VM_OPC_Log1pF32,
    VM_OPC_Log2F32,
    VM_OPC_PowF32,
    VM_OPC_RsqrtF32,
    VM_OPC_SqrtF32,
    VM_OPC_TanhF32,
    VM_OPC_ErfF32,

    VM_OPC_CmpEQF32O,
    VM_OPC_CmpEQF32U,
    VM_OPC_CmpNEF32O,
    VM_OPC_CmpNEF32U,
    VM_OPC_CmpLTF32O,
    VM_OPC_CmpLTF32U,
    VM_OPC_CmpLTEF32O,
    VM_OPC_CmpLTEF32U,
    VM_OPC_CmpNaNF32,

    VM_OPC_BufferLoadF32,
    VM_OPC_BufferStoreF32,
    VM_OPC_BufferFillF32,
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_F32
