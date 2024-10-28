// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_INTERFACES
#define IREE_DIALECT_ENCODING_INTERFACES

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def IREEEncoding_EncodingSolverInterfaceAttr :
  AttrInterface<"EncodingSolverInterfaceAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let description = [{
    Interface used to resolve encoding data from backends.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the materailized shape of the type.
      }],
      /*retTy=*/"::mlir::OpFoldResult",
      /*methodName=*/"calculateStorageElementCountInBytes",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "RankedTensorType":$type,
        "ValueRange":$dynamicDims
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the actual config that used in the solver. The config is
        expected to be wrapped into the encoding. Otherwise, we will duplicate
        the configuration in EncodingSolver and HAL::ExecutableTarget.
      }],
      /*retTy=*/"::mlir::DictionaryAttr",
      /*methodName=*/"getTargetConfig",
      /*args=*/(ins
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >
  ];
}

#endif // IREE_DIALECT_ENCODING_INTERFACES
