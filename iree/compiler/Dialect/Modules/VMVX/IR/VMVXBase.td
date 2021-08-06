// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_BASE
#define IREE_DIALECT_MODULES_VMVX_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE VMVX (Virtual Machine-based Linear Algebra) dialect
//===----------------------------------------------------------------------===//

def VMVX_Dialect : Dialect {
  let name = "vmvx";
  let cppNamespace = "::mlir::iree_compiler::IREE::VMVX";

  let summary = [{
    Vector extensions to the IREE VM.
  }];
  let description = [{
    This is a reference dialect representing a simple IREE VM-based linear
    algebra module that is used as a library at runtime. The ops in this dialect
    map (roughly) 1:1 with the exported functions in the runtime module.

    See `vmvx.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// VMVX enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VMVX types
//===----------------------------------------------------------------------===//

def VMVX_DeviceSize : TypeAlias<Index>;
def VMVX_DeviceSizeAttr : Util_IndexAttrBase<"size_t">;

def VMVX_HostSize : TypeAlias<Index>;
def VMVX_HostSizeAttr : Util_IndexAttrBase<"size_t">;

def VMVX_Index : TypeAlias<Index>;

def VMVX_HostBuffer : AnyTypeOf<[
  ByteBufferType,
  MutableByteBufferType,
]>;

def VMVX_Buffer : MemRefRankOf<[I8, I16, I32, I64, F32, F64], [0, 1]>;

//===----------------------------------------------------------------------===//
// VMVX op traits
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base VMVX op classes
//===----------------------------------------------------------------------===//

def VMVX_OpInterface : OpInterface<"VMVXOp"> {
  let description = [{
    Interface for VMVX ops that can be used to customize the lowering.
    This is required as there is not a way to get reflection information about
    ops.
  }];

  let methods = [
    StaticInterfaceMethod<
      "Extracts type information attributes that may be required by the op.",
      "void", "extractTypeAttributes", (ins "OperationState &":$state, "ArrayRef<Type>":$operandTypes, "ArrayRef<Type>":$resultTypes), [{
        ConcreteOp::extractTypeAttributes(state, operandTypes, resultTypes);
      }], [{
        return;  // default to no extraction
      }]
    >,
  ];
}

class VMVX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<VMVX_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_VMVX_BASE
