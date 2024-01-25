// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def IREEGPU_MmaArrayAttr : TypedArrayAttrBase<IREEGPU_MmaAttr,
    "Descriptor array of a list of supported mma operations">;

class IREEGPU_MmaVectorLayoutAttr<string attrname, string mmatype> :
    AttrDef<IREEGPU_Dialect, attrname, [
  DeclareAttrInterfaceMethods<IREEGPU_MmaAttr,[
    "getABCElementTypes",
    "getContractionLayout",
    "getMNKShape",
  ]>
]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  string baseDescription = [{
    Attribute describing a particular shape of matrix-multiply and accumulate
    instruction. Abstractly, all operations of this type represent the following
    unit of arithmetic for matrices A, B, and C.

    ```
      C += A x B
    ```

    Where the shape of matrix `A` is `[m, k]`, `B` is `[k, n]`, and
    `C` is `[m, n]`. This intentionally leaves the layout information abstract
    and uses interface methods to materialize layout information only when
    needed. The shape of the mma is stored explicitly here as that information
    is queried frequently.

    The element types for this particular mma type are |aType|, |bType|, and
    |cType| for matrices `A`, `B`, and `C` respectively.

    ######

  }];

    
  let parameters = (ins
    mmatype:$type,
    "int64_t":$mSize,
    "int64_t":$nSize,
    "int64_t":$kSize,
    "::mlir::Type":$aType,
    "::mlir::Type":$bType,
    "::mlir::Type":$cType
  );

  let extraClassDeclaration = [{
    ::std::tuple<Type, Type, Type> getABCElementTypes() {
      return ::std::make_tuple(getAType(), getBType(), getCType());
    }

    ::std::tuple<int64_t, int64_t, int64_t> getMNKShape() {
      return ::std::make_tuple(getMSize(), getNSize(), getKSize());
    }
  }];
}

class IREEGPU_I32MmaEnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

class IREEGPU_MmaEnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREEGPU_Dialect, enumInfo, name>;

//===----------------------------------------------------------------------===//
// MFMA Attributes
//===----------------------------------------------------------------------===//

def F16_16x16x16_F32 : I32EnumAttrCase<"F16_16x16x16_F32", 0>;
def F16_32x32x8_F32 : I32EnumAttrCase<"F16_32x32x8_F32", 1>;

def IREEGPU_MFMAType : IREEGPU_I32MmaEnumAttr<"MFMAType",
    "Descriptor for different MFMA operations", [
      F16_16x16x16_F32,
      F16_32x32x8_F32,
    ]>; 

def IREEGPU_MFMATypeAttr : IREEGPU_MmaEnumAttr<IREEGPU_MFMAType, "mfma_type">;

def IREEGPU_MFMA : IREEGPU_MmaVectorLayoutAttr<"MFMA", "MFMATypeAttr"> {
  let mnemonic = "mfma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = !strconcat(baseDescription, [{
    This mma variant describes configurations for MFMA ops. The |type| field
    specifies which particular MFMA type this refers to, with each type
    implicating a specific MNK shape and attribute type. The type enum name
    describes these fields as

    <InputType>_MxNxK_<CType>

    Where the element type for the `A` and `B` matrices are both `InputType`.
  }]);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "MFMAType":$type)>
  ];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS
