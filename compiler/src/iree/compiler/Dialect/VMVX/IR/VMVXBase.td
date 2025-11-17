// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_BASE
#define IREE_DIALECT_MODULES_VMVX_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"

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
  Util_BufferType,
]>;

def VMVX_ElementType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;
def VMVX_ElementTypeAttr : TypeAttrOf<VMVX_ElementType>;

// A potentially non-contiguous buffer of unknown providence.
def VMVX_NonContiguousBuffer : RankedOrUnrankedMemRefOf<
    [I8, I16, I32, I64, F32, F64]>;

def VMVX_Buffer : AnyTypeOf<[
  Util_BufferType,
]>;

//===----------------------------------------------------------------------===//
// VMVX op traits
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base VMVX op classes
//===----------------------------------------------------------------------===//

class VMVX_Op<string mnemonic, list<Trait> traits = []> :
    Op<VMVX_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_VMVX_BASE
