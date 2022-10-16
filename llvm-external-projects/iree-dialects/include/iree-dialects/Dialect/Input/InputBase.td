// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_BASE_TD
#define IREE_DIALECTS_DIALECT_INPUT_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREEInput_Dialect : Dialect {
  let name = "iree_input";
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
  let cppNamespace = "::mlir::iree_compiler::IREE::Input";
  let useDefaultTypePrinterParser = 1;
}

class IREEInput_Op<string mnemonic, list<Trait> traits = []> :
    Op<IREEInput_Dialect, mnemonic, traits>;
class IREEInput_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEInput_Dialect, mnemonic, !listconcat(traits, [Pure])>;
class IREEInput_Type<string name> : TypeDef<IREEInput_Dialect, name>;

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

class IREEInput_AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                                  "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let valueType = NoneType;
  let constBuilderCall = "mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
}
class IREEInput_AnyPtrOf<list<Type> types> :
    Type<And<[
      CPred<"$_self.isa<::mlir::iree_compiler::IREE::Input::PtrType>()">,
      Or<!foreach(type, types,
          SubstLeaves<
              "$_self",
              "$_self.cast<::mlir::iree_compiler::IREE::Input::PtrType>().getTargetType()",
              type.predicate>)>,
    ]>, !interleave(!foreach(type, types, type.summary), " or ")> {
  string builderCall = "";
}

def IREEInput_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;
def IREEInput_Tensor : TypeAlias<AnyRankedTensor>;

def IREEInput_AnyList : DialectType<
    IREEInput_Dialect,
    CPred<"$_self.isa<::mlir::iree_compiler::IREE::Input::ListType>()">,
      "list"> {
  let description = [{
    A mutable, resizable list of some type.
  }];
}

class IREEInput_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<::mlir::iree_compiler::IREE::Input::ListType>()">,
      SubstLeaves<"$_self",
                  "$_self.cast<::mlir::iree_compiler::IREE::Input::ListType>().getElementType()",
                  type.predicate>
    ]>, "list<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "::mlir::iree_compiler::IREE::Input::ListType::get(" # type.builderCall # ")");
}

def IREEInput_ElementTypeParameter : TypeParameter<
    "::mlir::Type", "A type suitable as an element type of a container">;
def IREEInput_PtrTargetTypeParameter : TypeParameter<
    "::mlir::Type", "A type suitable as a target type of a pointer">;

def IREEInput_Dim : TypeAlias<Index>;
def IREEInput_Dims : Variadic<IREEInput_Dim>;
def IREEInput_Shape : Variadic<IREEInput_Dim>;
def IREEInput_ShapeDynamicDims : Variadic<IREEInput_Dim>;

def IREEInput_GlobalRefAttr : IREEInput_AliasedSymbolRefAttr;
def IREEInput_AnyGlobalPtr : IREEInput_AnyPtrOf<[IREEInput_Tensor, IREEInput_PrimitiveType]>;

class IREEInput_IndexAttrBase<string descr> :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[
        CPred<"$_self.isa<IntegerAttr>()">,
        CPred<"$_self.cast<IntegerAttr>().getType().isIndex()">,
      ]>,
      descr> {
  let returnType = [{ APInt }];
}
def IREEInput_IndexAttr : IREEInput_IndexAttrBase<"size_t">;

def IREEInput_TiedOpStorageAttr :
    TypedArrayAttrBase<IREEInput_IndexAttr, "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}

#endif // IREE_DIALECTS_DIALECT_INPUT_BASE_TD
