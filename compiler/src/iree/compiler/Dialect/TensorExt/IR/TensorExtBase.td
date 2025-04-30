// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOREXT_BASE
#define IREE_DIALECT_TENSOREXT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

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
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// TensorExt dialect types
//===----------------------------------------------------------------------===//

def IREETensorExt_Dim : TypeAlias<Index>;
def IREETensorExt_ShapeDynamicDims : Variadic<IREETensorExt_Dim>;

//===----------------------------------------------------------------------===//
// Dispatch types
//===----------------------------------------------------------------------===//

def IREETensorExt_TensorAccessCanReadPred : CPred<[{
  cast<IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      IREE::TensorExt::TensorAccess::ReadOnly ||
  cast<IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      IREE::TensorExt::TensorAccess::ReadWrite
}]>;

def IREETensorExt_TensorAccessCanWritePred : CPred<[{
  cast<IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      IREE::TensorExt::TensorAccess::WriteOnly ||
  cast<IREE::TensorExt::DispatchTensorType>($_self).getAccess() ==
      IREE::TensorExt::TensorAccess::ReadWrite
}]>;

def IREETensorExt_DispatchTensor : DialectType<
    IREETensorExt_Dialect,
    CPred<"isa<IREE::TensorExt::DispatchTensorType>($_self)">,
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
      CPred<"isa<IREE::TensorExt::DispatchTensorType>($_self)">,
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
      CPred<"isa<IREE::TensorExt::DispatchTensorType>($_self)">,
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
