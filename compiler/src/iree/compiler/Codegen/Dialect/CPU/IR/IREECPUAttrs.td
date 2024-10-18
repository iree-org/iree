// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS

include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.td"
include "iree/compiler/Dialect/Encoding/IR/EncodingInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// iree_cpu.encoding_solver
//===----------------------------------------------------------------------===//

def IREECPU_VMVXEncodingSolverAttr :
    AttrDef<IREECPU_Dialect, "VMVXEncodingSolver", [
  DeclareAttrInterfaceMethods<IREEEncoding_EncodingSolverInterfaceAttr, [
    "calculateStorageElementCountInBytes",
  ]>
]> {
  let mnemonic = "vmvx_encoding_solver";
  let summary = "The encoding solver for VMVX backend";

  let assemblyFormat = "`<` struct(params) `>`";

  // The target configuration (from HAL::ExecutableTargetAttr) needs to be
  // initalized. Otherwise, it is not able to resolve encodings.
  let parameters = (ins
    AttrParameter<"DictionaryAttr", "">:$target_configuration
  );
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS
