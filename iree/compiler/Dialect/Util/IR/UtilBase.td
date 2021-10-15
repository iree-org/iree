// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_BASE
#define IREE_DIALECT_UTIL_IR_UTIL_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE base dialect used for types common across IREE subdialects.
//===----------------------------------------------------------------------===//

def Util_Dialect : Dialect {
  let name = "util";
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let summary = [{
    A dialect used for types common across IREE subdialects.
  }];
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def Util_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def Util_Element : AnyTypeOf<[AnySignlessInteger, AnyFloat]>;
def Util_MemRef : MemRefOf<[Util_Element]>;
def Util_Tensor : TensorOf<[Util_Element]>;

def Util_Offset : AnyTypeOf<[Index, AnyInteger]>;

class Util_IndexAttrBase<string descr> :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[
        CPred<"$_self.isa<IntegerAttr>()">,
        CPred<"$_self.cast<IntegerAttr>().getType().isIndex()">,
      ]>,
      descr> {
  let returnType = [{ APInt }];
}
def Util_IndexAttr : Util_IndexAttrBase<"size_t">;

def Util_TiedOpStorageAttr :
    TypedArrayAttrBase<Util_IndexAttr, "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}

//===----------------------------------------------------------------------===//
// Status codes
//===----------------------------------------------------------------------===//

def Util_Status : I<32>;

//===----------------------------------------------------------------------===//
// Attribute constraints
//===----------------------------------------------------------------------===//

def FuncTypeAttr : TypeAttrBase<"FunctionType", "function type">;

class IntegerAttrInRange<int min, int max> : AttrConstraint<
    CPred<"$_self.cast<IntegerAttr>().getInt() >= " # min # " && "
          "$_self.cast<IntegerAttr>().getInt() <= " # max>,
    "within the range [" # min # ", " # max # "] inclusive">;


class AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                        "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let valueType = NoneType;
  let constBuilderCall = "mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
}

//===----------------------------------------------------------------------===//
// Util_NamedTuple type constraint
//===----------------------------------------------------------------------===//

class Util_NamedTupleElement<int thisOrdinal, string thisName, Type thisType> {
  int ordinal = thisOrdinal;
  string name = thisName;
  Type type = thisType;
}

// Check if all of the tuple elements are buildable.
class Util_AreTupleElementsBuildable<list<Util_NamedTupleElement> lst> {
  bit ret = !foldl(1, lst, lhs, y, !and(lhs, !ne(y.type.builderCall, "")));
}

class Util_NamedTupleOf<list<Util_NamedTupleElement> elements, string descr> :
    Type<And<[
      IsTupleTypePred,
      CPred<"$_self.cast<TupleType>().size() == " # !size(elements)>,
      And<!foreach(element, elements,
          SubstLeaves<
              "$_self",
              "$_self.cast<TupleType>().getType(" # element.ordinal # ")",
              element.type.predicate>)>
    ]>, descr> {
  // Generate a builder call if all of the elements are buildable.
  let builderCall = !if(!eq(Util_AreTupleElementsBuildable<elements>.ret, 0),
    "", "$_builder.getTupleType({" #
        !interleave(!foreach(t, elements, t.type.builderCall), ", ") # "})"
  );
}

//===----------------------------------------------------------------------===//
// Util_StructAttr
//===----------------------------------------------------------------------===//
// This has a custom tablegen generator in StructAttrGen.cpp to create the
// attribute and storage types. It differs from the core MLIR StructAttr
// by more closely matching what handwritten C++ would have (better typing
// and ergonomics and custom parser/printer).

class Util_StructFieldAttr<string thisName, Attr thisType> {
  string name = thisName;
  Attr type = thisType;
}

class Util_StructAttr<string thisKind, string name, Dialect dialect,
                      list<Util_StructFieldAttr> attributes>
    : Attr<CPred<"$_self.isa<" # name # ">()">,
                 "structured attribute of " # name> {
  string kind = thisKind;
  string className = name;
  string cppNamespace = ?;
  let storageType = name;
  let returnType = name;
  let convertFromStorage = "$_self";
  Dialect structDialect = dialect;
  list<Util_StructFieldAttr> fields = attributes;
}

//===----------------------------------------------------------------------===//
// util.ptr<T> types
//===----------------------------------------------------------------------===//

def Util_AnyPtr : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::Util::PtrType>()">,
    "ptr"> {
  let description = [{
    Pointer to a typed value.
  }];
  let builderCall = "$_builder.getType<IREE::Util::PtrType>()";
}

class Util_PtrOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::Util::PtrType>()">,
      SubstLeaves<"$_self", "$_self.cast<IREE::Util::PtrType>().getTargetType()",
                  type.predicate>
    ]>, "ptr<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::Util::PtrType::get(" # type.builderCall # ")");
}

class Util_AnyPtrOf<list<Type> types> :
    Type<And<[
      CPred<"$_self.isa<IREE::Util::PtrType>()">,
      Or<!foreach(type, types,
          SubstLeaves<
              "$_self",
              "$_self.cast<IREE::Util::PtrType>().getTargetType()",
              type.predicate>)>,
    ]>, !interleave(!foreach(type, types, type.summary), " or ")> {
  string builderCall = "";
}

//===----------------------------------------------------------------------===//
// Global types
//===----------------------------------------------------------------------===//

def Util_GlobalRefAttr : Confined<FlatSymbolRefAttr, [
  ReferToOp<"IREE::Util::GlobalOp">,
]>;

def Util_AnyGlobalPtr : TypeAlias<Util_AnyPtr>;
// TODO(benvanik): actually implement Util_GlobalPtrOf.
class Util_GlobalPtrOf<list<Type> types> : TypeAlias<Util_AnyPtr>;

//===----------------------------------------------------------------------===//
// Buffer types
//===----------------------------------------------------------------------===//

def Util_AnySerializableAttr : Attr<Or<[
  CPred<"$_self.isa<mlir::DenseElementsAttr>()">,
  CPred<"$_self.isa<IREE::Util::SerializableAttrInterface>()">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ ::mlir::Attribute }];
  let returnType = [{ ::mlir::Attribute }];
  let convertFromStorage = "$_self";
}

// TODO(benvanik): rework these as a single type with an access mode attr.

def ByteBufferType : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::Util::ByteBufferType>()">,
    "byte_buffer"> {
  let description = [{
    A constant buffer of mapped host memory.
  }];
}
def ByteBufferRefAttr : AliasedSymbolRefAttr;

def MutableByteBufferType : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::Util::MutableByteBufferType>()">,
    "mutable_byte_buffer"> {
  let description = [{
    A buffer of read-write host memory.
  }];
}

//===----------------------------------------------------------------------===//
// List types
//===----------------------------------------------------------------------===//

def Util_AnyList : DialectType<
    Util_Dialect,
    CPred<"$_self.isa<IREE::Util::ListType>()">,
    "list"> {
  let description = [{
    A resizable list of some type.
  }];
}

class Util_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::Util::ListType>()">,
      SubstLeaves<"$_self",
                  "$_self.cast<IREE::Util::ListType>().getElementType()",
                  type.predicate>
    ]>, "list<" # type.summary # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::Util::ListType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

// TODO(b/143184519): add descriptions to other types.
// Operations with this trait indicate that they may yield execution of the
// current fiber. Invokers of the op must be yield-safe and assume that there
// may be a significant delay between when the operation is invoked and when it
// completes.
//
// TODO(benvanik): implement yield semantics.
// Need to add a new call type and function attr for 'async', then can validate
// entire parts of the call tree are either async-compatible or async-hostile.
// Only vm.call_async callees may contain an operation with YieldPoint.
def YieldPoint : NativeOpTrait<"IREE::Util::YieldPoint">;

// Denotes that an operation is potentially "unsafe" if used.
// Untrusted modules containing ops marked as unsafe will fail to verify at
// runtime if loaded for dynamic execution.
def Unsafe : NativeOpTrait<"IREE::Util::Unsafe">;

#endif  // IREE_DIALECT_UTIL_IR_UTIL_BASE
