// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "mlir/IR/EnumAttr.td"

def IREEGPU_MmaAttr : AttrInterface<"MmaAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Interface used to query information needed to generate code targeting a
    specific matrix multiplication intrinsic/instruction.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the shape of the mma operation described by this attribute.
        ```
          C += A * B
        ```
        Where, up to some transpositions, `A` has shape `<M, K>`, `B` has shape
        `<K, N>`, and `C` has shape `<M, N>`.
      }],
      /*retTy=*/"::std::tuple<int64_t, int64_t, int64_t>",
      /*methodName=*/"getMNKShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the element types of the operands to the mma operation
        ```
          C += A * B
        ```
      }],
      /*retTy=*/"::std::tuple<::mlir::Type, ::mlir::Type, ::mlir::Type>",
      /*methodName=*/"getABCElementTypes",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the vector layout for the A matrix of the mma operation
        ```
          C += A * B
        ```
      }],
      /*retTy=*/"::mlir::FailureOr<::std::tuple<"
                  "::mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface, "
                  "::mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface, "
                  "::mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface>>",
      /*methodName=*/"getContractionLayout",
      /*args=*/(ins "::mlir::vector::ContractionOp":$contract),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return failure();
      }]
    >
  ];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
