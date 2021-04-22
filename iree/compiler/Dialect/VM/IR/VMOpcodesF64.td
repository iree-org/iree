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
def VM_OPC_RemF64                : VM_OPC<0x27, "RemF64">;
def VM_OPC_AbsF64                : VM_OPC<0x2D, "AbsF64">;
def VM_OPC_NegF64                : VM_OPC<0x2E, "NegF64">;
def VM_OPC_CeilF64               : VM_OPC<0x2F, "CeilF64">;
def VM_OPC_FloorF64              : VM_OPC<0x31, "FloorF64">;
def VM_OPC_TruncF64F32           : VM_OPC<0x32, "TruncF64F32">;
def VM_OPC_ExtF32F64             : VM_OPC<0x37, "ExtF32F64">;
def VM_OPC_CmpEQF64              : VM_OPC<0x40, "CmpEQF64">;
def VM_OPC_CmpNEF64              : VM_OPC<0x41, "CmpNEF64">;
def VM_OPC_CmpNZF64              : VM_OPC<0x42, "CmpNZF64">;
def VM_OPC_CmpLTF64              : VM_OPC<0x43, "CmpLTF64">;

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
    VM_OPC_CmpEQF64,
    VM_OPC_CmpNEF64,
    VM_OPC_CmpNZF64,
    VM_OPC_CmpLTF64,
  ]>;

#endif  // IREE_DIALECT_VM_OPCODES_F64
