// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_BASE
#define IREE_DIALECT_UTIL_IR_UTIL_BASE

include "iree/compiler/Utils/DocMetadata.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// IREE base dialect used for types common across IREE subdialects
//===----------------------------------------------------------------------===//

def Util_Dialect : Dialect {
  let name = "util";
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";

  let summary = [{
    A dialect used for types common across IREE subdialects.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def Util_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def Util_Element : AnyTypeOf<[AnySignlessInteger, AnyFloat]>;
def Util_Tensor : TensorOf<[Util_Element]>;
def Util_Primitive : AnyTypeOf<[Index, AnyInteger, AnyFloat]>;

def Util_FillPattern : AnyTypeOf<[AnyInteger, AnyFloat, Index]>;

def Util_Offset : TypeAlias<Index>;
def Util_Size : TypeAlias<Index>;

def Util_Range : AnyTypeOf<[Index, AnyInteger]>;

// TODO(benvanik): a custom status type that allows for round-tripping full
// status objects.
def Util_Status : I<32>;

class Util_IndexAttrBase<string descr> :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[
        CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
        CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType().isIndex()">,
      ]>,
      descr> {
  let returnType = [{ APInt }];
}
def Util_IndexAttr : Util_IndexAttrBase<"size_t">;

def Util_TiedOpStorageAttr :
    TypedArrayAttrBase<Util_IndexAttr, "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}

defvar Util_GlobalRefAttr = FlatSymbolRefAttr;

def Util_AnySerializableAttr : Attr<Or<[
  CPred<"::llvm::isa<::mlir::DenseElementsAttr>($_self)">,
  CPred<"::llvm::isa<::mlir::DenseResourceElementsAttr>($_self)">,
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Util::SerializableAttrInterface>($_self)">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ ::mlir::Attribute }];
  let returnType = [{ ::mlir::Attribute }];
  let convertFromStorage = "$_self";
}

def Util_AnySerializableArrayAttr :
  TypedArrayAttrBase<Util_AnySerializableAttr,
                     "array attribute of serializable attributes"
  >;

class Util_AliasedSymbolRefAttr : Attr<CPred<"::llvm::isa<::mlir::FlatSymbolRefAttr>($_self)">,
                                       "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let valueType = NoneType;
  let constBuilderCall = "mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
}

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

// TODO(benvanik): implement yield semantics.
// Need to add a new call type and function attr for 'async', then can validate
// entire parts of the call tree are either async-compatible or async-hostile.
// Only vm.call_async callees may contain an operation with YieldPoint.
def Util_YieldPoint : NativeOpTrait<"IREE::Util::YieldPoint">;

// Denotes that an operation is potentially "unsafe" if used.
// Untrusted modules containing ops marked as unsafe will fail to verify at
// runtime if loaded for dynamic execution.
def Util_Unsafe : NativeOpTrait<"IREE::Util::Unsafe">;

// Denotes that an operation is only valid in debug builds.
// These ops are removed by the --iree-util-strip-debug-ops pass.
def Util_DebugOnly : NativeOpTrait<"IREE::Util::DebugOnly">;

// Denotes an op that represents an object-like declaration (class/table/etc).
// Must be a symbol table (containing members) and isolated from above
// (instantiatable and only referenced by symbols instead of SSA values).
def Util_ObjectLike : NativeOpTrait<"IREE::Util::ObjectLike">;

// Denotes an op that marks unreachable code at runtime.
// Used for both terminator (util.unreachable) and non-terminator
// (util.scf.unreachable) unreachable operations.
def Util_UnreachableLike : NativeOpTrait<"IREE::Util::UnreachableLike">;

//===----------------------------------------------------------------------===//
// Base util dialect type/attr/interface classes with doc metadata
//===----------------------------------------------------------------------===//

class Util_Op<string mnemonic, list<Trait> traits = []>
    : Op<Util_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;
}

class Util_PureOp<string mnemonic, list<Trait> traits = []> :
    Util_Op<mnemonic, !listconcat(traits, [Pure])>;

class Util_TypeDef<string name, list<Trait> traits = []> :
    TypeDef<Util_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class Util_AttrDef<string name, list<Trait> traits = []> :
    AttrDef<Util_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

class Util_OpInterface<string name, list<Interface> baseInterfaces = []> :
    OpInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class Util_TypeInterface<string name, list<Interface> baseInterfaces = []> :
    TypeInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class Util_AttrInterface<string name, list<Interface> baseInterfaces = []> :
    AttrInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Util";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

//===----------------------------------------------------------------------===//
// Util Dialect TableGen File Structure and Dependencies
//===----------------------------------------------------------------------===//
//
// This dialect's definitions are partitioned into the following files
// to enhance modularity and manage dependencies:
//
// 1. UtilBase.td: Defines the core 'util' dialect and fundamental base
//    classes for ops, types, attributes, and interfaces, along with common
//    type aliases and op traits.
//
// 2. UtilAttrs.td: Defines all dialect-specific attributes.
//    Includes: UtilBase.td (for Util_AttrDef).
//
// 3. UtilInterfaces.td: Defines all dialect-specific interfaces.
//    Includes: UtilBase.td (for base interface classes).
//
// 4. UtilTypes.td: Defines all dialect-specific types.
//    Includes: UtilBase.td (for Util_TypeDef),
//              UtilInterfaces.td (for types implementing interfaces).
//
// 5. UtilOps.td: Defines all dialect-specific operations.
//    Includes: UtilBase.td (for Util_Op and traits),
//              UtilInterfaces.td (for ops implementing interfaces),
//              UtilTypes.td (for types used in ops).
//
// This structure aims to prevent circular dependencies and improve
// maintainability.

#endif  // IREE_DIALECT_UTIL_IR_UTIL_BASE
