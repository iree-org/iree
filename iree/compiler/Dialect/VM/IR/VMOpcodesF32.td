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
def VM_OPC_ConstF32Zero          : VM_OPC<0x08, "ConstF32Zero">;
def VM_OPC_ConstF32              : VM_OPC<0x09, "ConstF32">;
def VM_OPC_ListGetF32            : VM_OPC<0x14, "ListGetF32">;
def VM_OPC_ListSetF32            : VM_OPC<0x15, "ListSetF32">;
def VM_OPC_SelectF32             : VM_OPC<0x1E, "SelectF32">;
def VM_OPC_SwitchF32             : VM_OPC<0x20, "SwitchF32">;
def VM_OPC_AddF32                : VM_OPC<0x22, "AddF32">;
def VM_OPC_SubF32                : VM_OPC<0x23, "SubF32">;
def VM_OPC_MulF32                : VM_OPC<0x24, "MulF32">;
def VM_OPC_DivF32                : VM_OPC<0x25, "DivF32">;
def VM_OPC_RemF32                : VM_OPC<0x27, "RemF32">;
def VM_OPC_AbsF32                : VM_OPC<0x2D, "AbsF32">;
def VM_OPC_NegF32                : VM_OPC<0x2E, "NegF32">;
def VM_OPC_CeilF32               : VM_OPC<0x2F, "CeilF32">;
def VM_OPC_FloorF32              : VM_OPC<0x32, "FloorF32">;
def VM_OPC_CmpEQF32              : VM_OPC<0x40, "CmpEQF32">;
def VM_OPC_CmpNEF32              : VM_OPC<0x41, "CmpNEF32">;
def VM_OPC_CmpNZF32              : VM_OPC<0x42, "CmpNZF32">;
def VM_OPC_CmpLTF32              : VM_OPC<0x43, "CmpLTF32">;

def VM_OPC_BufferLoadF32         : VM_OPC<0xB0, "BufferLoadF32">;
def VM_OPC_BufferStoreF32        : VM_OPC<0xB1, "BufferStoreF32">;
def VM_OPC_BufferFillF32         : VM_OPC<0xC0, "BufferFillF32">;

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
    VM_OPC_AbsF32,
    VM_OPC_NegF32,
    VM_OPC_CeilF32,
    VM_OPC_FloorF32,
    VM_OPC_CmpEQF32,
    VM_OPC_CmpNEF32,
    VM_OPC_CmpNZF32,
    VM_OPC_CmpLTF32,

    VM_OPC_BufferLoadF32,
    VM_OPC_BufferStoreF32,
    VM_OPC_BufferFillF32,
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_F32
