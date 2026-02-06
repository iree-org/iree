// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_BASE
#define IREE_DIALECT_STREAM_BASE

include "iree/compiler/Dialect/Stream/IR/StreamDialect.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base dialect op/type/attr/interface classes
//===----------------------------------------------------------------------===//

class Stream_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stream_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;
}

class Stream_PureOp<string mnemonic, list<Trait> traits = []> :
    Stream_Op<mnemonic, !listconcat(traits, [Pure])>;

class Stream_TypeDef<string name, list<Trait> traits = []> :
    TypeDef<Stream_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class Stream_AttrDef<string name, list<Trait> traits = []> :
    AttrDef<Stream_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

class Stream_OpInterface<string name, list<Interface> baseInterfaces = []> :
    OpInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class Stream_TypeInterface<string name, list<Interface> baseInterfaces = []> :
    TypeInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class Stream_AttrInterface<string name, list<Interface> baseInterfaces = []> :
    AttrInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

//===----------------------------------------------------------------------===//
// Stream dialect type aliases
//===----------------------------------------------------------------------===//

def Stream_PrimitiveType : AnyTypeOf<[Index, AnyInteger, AnyFloat, AnyComplex]>;
def Stream_FillPatternType : AnyTypeOf<[I8, I16, I32, I64]>;
def Stream_NativeFillPatternType : AnyTypeOf<[I8, I16, I32]>;

def Stream_Offset : TypeAlias<Index>;
def Stream_Size : TypeAlias<Index>;

def Stream_Tensor : TypeAlias<AnyRankedTensor>;
def Stream_Dim : TypeAlias<Index>;
def Stream_ShapeDynamicDims : Variadic<Stream_Dim>;

def Stream_IndexAttr : Util_IndexAttrBase<"index">;
def Stream_IndexArrayAttr : TypedArrayAttrBase<Stream_IndexAttr,
                                               "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def Stream_ExecutableRefAttr : Util_AliasedSymbolRefAttr;
def Stream_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def Stream_GlobalPtr : Util_AnyPtrOf<[Stream_Tensor, Stream_PrimitiveType]>;

//===----------------------------------------------------------------------===//
// Stream op traits
//===----------------------------------------------------------------------===//

def Stream_TensorPhaseOp : NativeOpTrait<"IREE::Stream::TensorPhaseOp">;
def Stream_AsyncPhaseOp : NativeOpTrait<"IREE::Stream::AsyncPhaseOp">;
def Stream_CmdPhaseOp : NativeOpTrait<"IREE::Stream::CmdPhaseOp">;
def Stream_TiedResourcePassthrough
    : NativeOpTrait<"IREE::Stream::TiedResourcePassthrough">;

#endif  // IREE_DIALECT_STREAM_BASE
