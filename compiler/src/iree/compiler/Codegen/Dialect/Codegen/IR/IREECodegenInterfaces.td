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
        Returns true if the lowering config specifies tile sizes for the given
        tiling level.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasTilingLevel",
      /*args=*/(ins "unsigned":$level),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the lowering config specifies tile sizes for the
        workgroup tiling level.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasWorkgroupTilingLevel",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
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
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the name of the custom lowering strategy to apply to the annotated
        operation.
      }],
      /*retTy=*/"::std::optional<::llvm::StringRef>",
      /*methodName=*/"getLoweringStrategy",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >
  ];
}

def IREECodegen_LayoutAttrInterface :
  AttrInterface<"LayoutAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    An interface that collects a set of methods for encoding materialization.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the layout of materialized encoding for a tensor type.
      }],
      /*retTy=*/"::mlir::iree_compiler::IREE::Codegen::MaterializeEncodingInfo",
      /*methodName=*/"getEncodingInfo",
      /*args=*/(ins "::mlir::RankedTensorType":$type),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return MaterializeEncodingInfo{};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the layout of materialized encoding for a tensor type.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"lowerOp",
      /*args=*/(ins "::mlir::OpBuilder &":$b,
                    "::mlir::Operation *":$op,
                    "::mlir::TypeRange":$convertedResTypes,
                    "::mlir::ValueRange":$convertedOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return nullptr;
      }]
    >
  ];
}

def IREECodegen_SwizzleAttrInterface :
  AttrInterface<"SwizzleAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    An interface that describes 1D memref swizzling patterns.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Swizzles |offset| into memref |src|.
      }],
      /*retTy=*/"::mlir::OpFoldResult",
      /*methodName=*/"swizzleOffset",
      /*args=*/(ins "::mlir::OpBuilder&":$b,
                    "::mlir::Location":$loc,
                    "::mlir::OpFoldResult":$offset,
                    "::mlir::Value":$src),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return offset;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of elements that remain contiguous with the swizzling
        pattern. This unrolls all accesses to this element count.

        Currently swizzling is only supported if all accesses are multiples of
        this value.

        TODO: Support non-width aligned swizzling.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getAccessElementCount",
      /*args=*/(ins)
    >
  ];
}

def IREECodegen_AnySwizzleAttr : Attr<Or<[
  CPred<"isa<IREE::Codegen::SwizzleAttrInterface>($_self)">,
]>, "swizzling descriptor attributes"> {
  let storageType = [{ IREE::Codegen::SwizzleAttrInterface }];
  let returnType = [{ IREE::Codegen::SwizzleAttrInterface }];
  let convertFromStorage = "$_self";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
