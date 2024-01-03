// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_BASE
#define IREE_DIALECT_UTIL_IR_UTIL_BASE

include "mlir/IR/EnumAttr.td"
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

defvar Util_GlobalRefAttr = FlatSymbolRefAttr;

def Util_AnySerializableAttr : Attr<Or<[
  CPred<"$_self.isa<mlir::DenseElementsAttr>()">,
  CPred<"$_self.isa<mlir::DenseResourceElementsAttr>()">,
  CPred<"$_self.isa<IREE::Util::SerializableAttrInterface>()">,
]>, "buffer-like constant attribute values"> {
  let storageType = [{ ::mlir::Attribute }];
  let returnType = [{ ::mlir::Attribute }];
  let convertFromStorage = "$_self";
}

def Util_AnySerializableArrayAttr :
  TypedArrayAttrBase<Util_AnySerializableAttr,
                     "array attribute of serializable attributes"
  >;

class Util_AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
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

#endif  // IREE_DIALECT_UTIL_IR_UTIL_BASE
