// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IRREE_CODEGEN_DIALECT_PROCESSOR_INTERFACES
#define IRREE_CODEGEN_DIALECT_PROCESSOR_INTERFACES

include "mlir/IR/OpBase.td"

def ProcessorIDInterface : OpInterface<"ProcessorIDInterface"> {
  let description = [{
    This interface indicates that the implementing op is a processor ID
    intrinsic and a provides unified way to query the ID dimension.
  }];
  let cppNamespace = "::mlir::iree_compiler";

  let methods = [
    InterfaceMethod<[{
        Returns the processor ID dimension index.
      }],
      "unsigned", "getDimIndex", (ins)
    >
  ];
}

def ProcessorCountInterface : OpInterface<"ProcessorCountInterface"> {
  let description = [{
    This interface indicates that the implementing op is a processor count
    intrinsic and a provides unified way to query the count dimension.
  }];
  let cppNamespace = "::mlir::iree_compiler";

  let methods = [
    InterfaceMethod<[{
        Returns the processor count dimension index.
      }],
      "unsigned", "getDimIndex", (ins)
    >
  ];
}

def ProcessorTileSizeInterface : OpInterface<"ProcessorTileSizeInterface"> {
  let description = [{
    This interface indicates that the implementing op is a processor tile
    size symbolic value and provides unified way to query the count dimension.
  }];
  let cppNamespace = "::mlir::iree_compiler";

  let methods = [
    InterfaceMethod<[{
        Returns the processor tile size dimension index.
      }],
      "unsigned", "getDimIndex", (ins)
    >
  ];
}

#endif // IRREE_CODEGEN_DIALECT_PROCESSOR_INTERFACES
