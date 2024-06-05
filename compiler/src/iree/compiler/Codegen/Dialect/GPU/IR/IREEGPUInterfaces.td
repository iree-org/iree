// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td"
include "mlir/IR/EnumAttr.td"

def IREEGPU_MmaInterfaceAttr : AttrInterface<"MmaInterfaceAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Interface used to query information needed to generate code targeting a
    specific Matrix Multiply–Accumulate (MMA) machine intrinsic instruction.
    Layout information can be explicit or opaque, however all such attributes
    must specify the static shape of the operation and the required element
    types.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the shape of the MMA operation:
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
        Returns the element types of the operands to the MMA operation:
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
        Returns the vector types of the operands to the MMA operation:
        ```
          C += A * B
        ```
        Each thread participating in the MMA operation is responsible for
        storing such shaped vectors in the registers.
      }],
      /*retTy=*/"::std::tuple<::mlir::VectorType, ::mlir::VectorType, ::mlir::VectorType>",
      /*methodName=*/"getABCVectorTypes",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the vector layout for the A/B/C matrix of the MMA operation:
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
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the expected subgroup size of the MMA operation/intrinsic.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getSubgroupSize",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Constructs a concrete instance of the associated mma operation.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::Value>",
      /*methodName=*/"buildMmaOperation",
      /*args=*/(ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::Location":$loc,
        "::mlir::Type":$result_type,
        "::mlir::Value":$lhs,
        "::mlir::Value":$rhs,
        "::mlir::Value":$acc
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return failure();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Constructs the offsets/sizes/strides for extracting the per-thread
        slice of the given operand fragment.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"populateOperandOffsetsSizesStrides",
      /*args=*/(ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::Location":$loc,
        "::mlir::iree_compiler::IREE::GPU::MMAFragment":$fragment,
        "::mlir::Value":$lane_id,
        "::llvm::ArrayRef<int64_t>":$permutation,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$offsets,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$sizes,
        "::llvm::SmallVector<::mlir::OpFoldResult>&":$strides
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return failure();
      }]
    >,
  ];
}

def IREEGPU_AnyMmaAttr : Attr<Or<[
  CPred<"isa<IREE::GPU::MmaInterfaceAttr>($_self)">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ IREE::GPU::MmaInterfaceAttr }];
  let returnType = [{ IREE::GPU::MmaInterfaceAttr }];
  let convertFromStorage = "$_self";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
