// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class IREEGPU_I32MmaEnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// MMA Attribute Enums
//===----------------------------------------------------------------------===//

def MFMA_F16_16x16x16_F32 : I32EnumAttrCase<"MFMA_F16_16x16x16_F32", 0>;
def MFMA_F16_32x32x8_F32 : I32EnumAttrCase<"MFMA_F16_32x32x8_F32", 1>;
def WMMA_F16_16x16x16_F32 : I32EnumAttrCase<"WMMA_F16_16x16x16_F32", 2>;

def IREEGPU_MMAIntrinsic : IREEGPU_I32MmaEnumAttr<"MMAIntrinsic",
    "Descriptor for different MMA intrinsics", [
      MFMA_F16_16x16x16_F32,
      MFMA_F16_32x32x8_F32,
      WMMA_F16_16x16x16_F32
    ]>;

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
