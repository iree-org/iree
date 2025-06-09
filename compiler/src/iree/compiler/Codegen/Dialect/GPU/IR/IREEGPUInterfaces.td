// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td"
include "mlir/IR/EnumAttr.td"

def IREEGPU_MmaInterfaceAttr
    : AttrInterface<"MmaInterfaceAttr", [IREECodegen_InnerTileDescAttrInterface]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Interface used to query information needed to generate code targeting a
    specific Matrix Multiply-Accumulate (MMA) machine intrinsic instruction.
    Layout information can be explicit or opaque, however all such attributes
    must specify the static shape of the operation and the required element
    types.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the subgroup size needed to execute this kind of multi-MMA.
      }],
      /*retType=*/"int64_t",
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
  ];
  // Methods included to preserve interfaces from beforet he InnerTileDescAttrInterface
  // refactoring.
  let extraSharedClassDeclaration = [{
    /// Returns the shape of the MMA operation:
    /// ```
    ///   C += A * B
    /// ```
    /// Where, up to some transpositions, `A` has shape `<M, K>`, `B` has shape
    /// `<K, N>`, and `C` has shape `<M, N>`.
    ::std::tuple<int64_t, int64_t, int64_t>
    getMNKShape() const {
      ::llvm::SmallVector<::mlir::VectorType> preThreadTypes;
      $_attr.getUndistributedTileTypes(preThreadTypes);
      ::llvm::ArrayRef<int64_t> accShape = preThreadTypes[2].getShape();
      ::llvm::ArrayRef<int64_t> lhsShape = preThreadTypes[0].getShape();
      return {accShape[0], accShape[1], lhsShape[1]};
    }

    /// Returns the element types of the operands to the MMA operation:
    /// ```
    ///   C += A * B
    /// ```
    ::std::tuple<::mlir::Type, ::mlir::Type, ::mlir::Type>
    getABCElementTypes() const {
      ::llvm::SmallVector<::mlir::Type> elementTypes;
      $_attr.getElementTypes(elementTypes);
      return {elementTypes[0], elementTypes[1], elementTypes[2]};
    }

    /// Returns the vector types of the operands to the MMA operation:
    /// ```
    /// C += A * B
    /// ```
    /// Each thread participating in the MMA operation is responsible for
    /// storing such shaped vectors in the registers.
    ::std::tuple<::mlir::VectorType, ::mlir::VectorType, ::mlir::VectorType>
    getABCVectorTypes() const {
      ::llvm::SmallVector<::mlir::VectorType> threadTypes;
      $_attr.getDistributedTileTypes(threadTypes);
      return {threadTypes[0], threadTypes[1], threadTypes[2]};
    }
  }];
}

def IREEGPU_AnyMmaAttr : Attr<Or<[
  CPred<"isa<IREE::GPU::MmaInterfaceAttr>($_self)">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ IREE::GPU::MmaInterfaceAttr }];
  let returnType = [{ IREE::GPU::MmaInterfaceAttr }];
  let convertFromStorage = "$_self";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES
