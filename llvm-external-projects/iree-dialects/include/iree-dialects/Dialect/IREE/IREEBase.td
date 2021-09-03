// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_BASE_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREE_Dialect : Dialect {
  let name = "iree";
  let summary = "Public ops/type/attributes legal for input to IREE's compiler";
  let description = [{
    IREE's compiler allows as input a number of common dialects. This dialect
    contains structural and unique ops that do not exist elsewhere or that IREE
    has an interest in maintaining as a stable set.

    The contents of this dialect often mirror various constructs in IREE's
    internal implementation. The focus here is on simplicity and stability
    over time. Generally, this dialect does not use "advanced" features and
    should be broadly source compatible over a range of LLVM versions. There
    are of course, limits, and source-compatibility is not guaranteed, since
    LLVM/MLIR's API surface is itself unstable.
  }];
  let cppNamespace = "::mlir::iree";
}

class IREE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits>;
class IREE_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])>;
class IREE_Type<string name> : TypeDef<IREE_Dialect, name>;

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

class IREE_AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                        "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let valueType = NoneType;
  let constBuilderCall = "mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
}
class IREE_AnyPtrOf<list<Type> types> :
    Type<And<[
      CPred<"$_self.isa<::mlir::iree::PtrType>()">,
      Or<!foreach(type, types,
          SubstLeaves<
              "$_self",
              "$_self.cast<::mlir::iree::PtrType>().getTargetType()",
              type.predicate>)>,
    ]>, !interleave(!foreach(type, types, type.summary), " or ")> {
  string builderCall = "";
}

def IREE_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;
def IREE_Tensor : TypeAlias<AnyRankedTensor>;

def IREE_AnyList : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<::mlir::iree::ListType>()">,
      "list"> {
  let description = [{
    A mutable, resizable list of some type.
  }];
}

class IREE_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<::mlir::iree::ListType>()">,
      SubstLeaves<"$_self",
                  "$_self.cast<::mlir::iree::ListType>().getElementType()",
                  type.predicate>
    ]>, "list<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "::mlir::iree::ListType::get(" # type.builderCall # ")");
}

def IREE_ElementTypeParameter : TypeParameter<
    "::mlir::Type", "A type suitable as an element type of a container">;
def IREE_PtrTargetTypeParameter : TypeParameter<
    "::mlir::Type", "A type suitable as a target type of a pointer">;

def IREE_Dim : TypeAlias<Index>;
def IREE_Dims : Variadic<IREE_Dim>;
def IREE_Shape : Variadic<IREE_Dim>;
def IREE_ShapeDynamicDims : Variadic<IREE_Dim>;

def IREE_GlobalRefAttr : IREE_AliasedSymbolRefAttr;
def IREE_AnyGlobalPtr : IREE_AnyPtrOf<[IREE_Tensor, IREE_PrimitiveType]>;

class IREE_IndexAttrBase<string descr> :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[
        CPred<"$_self.isa<IntegerAttr>()">,
        CPred<"$_self.cast<IntegerAttr>().getType().isIndex()">,
      ]>,
      descr> {
  let returnType = [{ APInt }];
}
def IREE_IndexAttr : IREE_IndexAttrBase<"size_t">;

def IREE_TiedOpStorageAttr :
    TypedArrayAttrBase<IREE_IndexAttr, "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_BASE_TD
