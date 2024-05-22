// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"

def IREECodegen_LoweringConfigAttrInterface :
  AttrInterface<"LoweringConfigAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    Attribute interface for specifying information used by the compiler to
    translate a specific operation. The way this information is used is backend
    and/or pipeline specific, so this interface only exposes information needed
    for shared use cases.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the tile sizes to use for workgroup distribution.

        TODO: This should be queried/constructed from device information and
        used to compute workgroup size.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupTileSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the loop interchange to use for workgroup distribution.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupInterchange",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the tile sizes for the specified tiling level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        returns an empty list if sizes are not specified for this level. dynamic
        sizes are specified with `shapedtype::kdynamicsize`.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getStaticTilingLevelSizes",
      /*args=*/(ins
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Constructs the tile sizes for the specified level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        Returns an empty list if sizes are not specified for this level.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTilingLevelSizes",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<OpFoldResult>();
      }]
    >
  ];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
