// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOREXT_BASE
#define IREE_DIALECT_TENSOREXT_BASE

include "iree/compiler/Utils/DocMetadata.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREETensorExt_Dialect : Dialect {
  let name = "iree_tensor_ext";
  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";
  let summary = [{
    IREE Tensor Extensions.
  }];
  let description = [{
    A dialect designed for experimenting with tensor operations
    beyond what is currently available in the Tensor Dialect.
  }];
  let dependentDialects = [
    "arith::ArithDialect" // For materializeConstant()
  ];
  let extraClassDeclaration = [{
    void initializeAttrs();
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// TensorExt dialect types
//===----------------------------------------------------------------------===//

def IREETensorExt_Dim : TypeAlias<Index>;
def IREETensorExt_ShapeDynamicDims : Variadic<IREETensorExt_Dim>;

//===----------------------------------------------------------------------===//
// Base dialect op/type/attr/interface classes
//===----------------------------------------------------------------------===//

class IREETensorExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<IREETensorExt_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;
}

class IREETensorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    IREETensorExt_Op<mnemonic, !listconcat(traits, [Pure])>;

class IREETensorExt_TypeDef<string name, list<Trait> traits = []> :
    TypeDef<IREETensorExt_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class IREETensorExt_AttrDef<string name, list<Trait> traits = []> :
    AttrDef<IREETensorExt_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

class IREETensorExt_OpInterface<string name, list<Interface> baseInterfaces = []> :
    OpInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class IREETensorExt_TypeInterface<string name, list<Interface> baseInterfaces = []> :
    TypeInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class IREETensorExt_AttrInterface<string name, list<Interface> baseInterfaces = []> :
    AttrInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

//===----------------------------------------------------------------------===//
// Dispatch types
//===----------------------------------------------------------------------===//

def IREETensorExt_TensorAccessCanReadPred : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      ::mlir::iree_compiler::IREE::TensorExt::TensorAccess::ReadOnly ||
  ::llvm::cast<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      ::mlir::iree_compiler::IREE::TensorExt::TensorAccess::ReadWrite
}]>;

def IREETensorExt_TensorAccessCanWritePred : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      ::mlir::iree_compiler::IREE::TensorExt::TensorAccess::WriteOnly ||
  ::llvm::cast<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      ::mlir::iree_compiler::IREE::TensorExt::TensorAccess::ReadWrite
}]>;

def IREETensorExt_DispatchTensor : DialectType<
    IREETensorExt_Dialect,
    CPred<"::llvm::isa<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self)">,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region input/output operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as both
    load and store from the backing tensor representation.
  }];
}

def IREETensorExt_ReadableDispatchTensor : DialectType<
    IREETensorExt_Dialect,
    And<[
      CPred<"::llvm::isa<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self)">,
      IREETensorExt_TensorAccessCanReadPred,
    ]>,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region input operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as load
    from the backing tensor representation.
  }];
}

def IREETensorExt_WritableDispatchTensor : DialectType<
    IREETensorExt_Dialect,
    And<[
      CPred<"::llvm::isa<::mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>($_self)">,
      IREETensorExt_TensorAccessCanWritePred,
    ]>,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region output operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as store
    to the backing tensor representation.
  }];
}

#endif // IREE_DIALECT_TENSOREXT_BASE
