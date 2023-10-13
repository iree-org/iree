// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_BASE_TD
#define IREE_DIALECTS_DIALECT_INPUT_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREEInput_Dialect : Dialect {
  let name = "iree_input";
  let summary = [{
    Public ops/type/attributes legal for input to IREE's compiler.
  }];
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
  let useDefaultAttributePrinterParser = 1;
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

def IREEInput_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat, AnyComplex]>;
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

//===----------------------------------------------------------------------===//
// Type aliases for working with shapes
//===----------------------------------------------------------------------===//

def IREEInput_Dim : TypeAlias<Index>;
def IREEInput_Dims : Variadic<IREEInput_Dim>;
def IREEInput_Shape : Variadic<IREEInput_Dim>;
def IREEInput_ShapeDynamicDims : Variadic<IREEInput_Dim>;

//===----------------------------------------------------------------------===//
// Type aliases for working with Buffers and BufferViews
//===----------------------------------------------------------------------===//

def IREEInput_DeviceSize : TypeAlias<Index>;
def IREEInput_ElementType : TypeAlias<I32>;
def IREEInput_EncodingType : TypeAlias<I32>;

//===----------------------------------------------------------------------===//
// Following attributes are mirrored from the IREEs HAL dialect to allow higher
// level clients of IREE to construct IREE input with pre-compiled executables,
// e.g. in NVIDIA GPU case it allows the higher level client to embed PTX or
// CUBIN as an executable, export device functions, and rely on IREEs runtime
// to dispatch them. It is possible to constuct an input IR with an arbitrary
// mix of pre-compiled kernels (executables) and "compute IR" (e.g. stablehlo)
// that will be compiled to executables using builtin IREE compilation pipeline.
//
// See corresponding HAL dialect attributes for the documentation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Device and executable target specification
//===----------------------------------------------------------------------===//

def IREEInput_DeviceTargetAttr : AttrDef<IREEInput_Dialect, "DeviceTarget"> {
  let mnemonic = "device.target";
  let summary = [{generic device target specification}];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$deviceID,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );

  let builders = [
    AttrBuilder<(ins "StringRef":$deviceID)>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def IREEInput_ExecutableTargetAttr :
    AttrDef<IREEInput_Dialect, "ExecutableTarget"> {
  let mnemonic = "executable.target";
  let summary = [{generic executable target specification}];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$backend,
    AttrParameter<"StringAttr", "">:$format,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );

  let builders = [
    AttrBuilder<(ins "StringRef":$backend, "StringRef":$format)>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Executable object(s)
//===----------------------------------------------------------------------===//

def IREEInput_WorkgroupSizeAttr : TypedArrayAttrBase<
    IREEInput_IndexAttrBase<"size_t">,
    "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def IREEInput_SubgroupSizeAttr : IREEInput_IndexAttrBase<"size_t">;

def IREEInput_ExecutableObjectAttr :
    AttrDef<IREEInput_Dialect, "ExecutableObject"> {
  let mnemonic = "executable.object";
  let summary = [{executable object reference}];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$path,
    OptionalParameter<"DenseIntElementsAttr", "">:$data
  );

  let hasCustomAssemblyFormat = 1;
}

def IREEInput_ExecutableObjectArrayAttr :
    TypedArrayAttrBase<IREEInput_ExecutableObjectAttr,
                       "IREEInput executable object references">;

def IREEInput_ExecutableObjectsAttr :
    AttrDef<IREEInput_Dialect, "ExecutableObjects"> {
  let mnemonic = "executable.objects";
  let summary = [{target-specific object file references}];

  let parameters = (ins
    AttrParameter<"ArrayAttr", "">:$targets,
    AttrParameter<"ArrayAttr", "">:$targetObjects
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Executable Pipeline Layout (aka IREEs executable ABI)
//===----------------------------------------------------------------------===//

def IREEInput_OrdinalAttr : IREEInput_IndexAttrBase<"size_t">;

class IREEInput_I32Enum<string name, string description,
                        list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::iree_compiler::IREE::Input";
}

class IREEInput_I32EnumAttr<string name, string description, string mnemonic,
                            list<I32EnumAttrCase> cases>
    : EnumAttr<IREEInput_Dialect,
               IREEInput_I32Enum<name, description, cases>, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def IREEInput_DescriptorType_UniformBuffer :
    I32EnumAttrCase<"UniformBuffer", 6, "uniform_buffer">;
def IREEInput_DescriptorType_StorageBuffer :
    I32EnumAttrCase<"StorageBuffer", 7, "storage_buffer">;
def IREEInput_DescriptorTypeAttr :
    IREEInput_I32EnumAttr<"DescriptorType", "valid DescriptorType",
                                            "descriptor_type", [
      IREEInput_DescriptorType_UniformBuffer,
      IREEInput_DescriptorType_StorageBuffer,
    ]>;

def IREEInput_DescriptorFlags_None :
    I32BitEnumAttrCase<"None", 0x0000>;
def IREEInput_DescriptorFlags_ReadOnly :
    I32BitEnumAttrCase<"ReadOnly", 0x0001>;
def IREEInput_DescriptorFlagsAttr :
    I32BitEnumAttr<"DescriptorFlags", "valid Descriptor flags", [
      IREEInput_DescriptorFlags_None,
      IREEInput_DescriptorFlags_ReadOnly,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Input";
}

def IREEInput_DescriptorSetBindingAttr :
    AttrDef<IREEInput_Dialect, "DescriptorSetBinding", []> {
  let mnemonic = "descriptor_set.binding";
  let summary = [{descriptor set binding specification}];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    AttrParameter<"DescriptorType", "">:$type,
    OptionalParameter<"std::optional<DescriptorFlags>">:$flags
  );

  let assemblyFormat = [{
    `<` $ordinal `,` $type (`,` $flags^)? `>`
  }];
}

def IREEInput_DescriptorSetLayoutFlags_None :
    I32BitEnumAttrCase<"None", 0x0000>;
def IREEInput_DescriptorSetLayoutFlags_Indirect :
    I32BitEnumAttrCase<"Indirect", 0x0001>;
def IREEInput_DescriptorSetLayoutFlagsAttr :
    I32BitEnumAttr<"DescriptorSetLayoutFlags", "valid DescriptorSetLayout flags", [
      IREEInput_DescriptorSetLayoutFlags_None,
      IREEInput_DescriptorSetLayoutFlags_Indirect,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Input";
}

def IREEInput_DescriptorSetLayoutAttr :
    AttrDef<IREEInput_Dialect, "DescriptorSetLayout", []> {
  let mnemonic = "descriptor_set.layout";
  let summary = [{descriptor set layout specification}];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    ArrayRefParameter<"DescriptorSetBindingAttr", "">:$bindings,
    OptionalParameter<"std::optional<DescriptorSetLayoutFlags>">:$flags
  );

  let assemblyFormat = [{
    `<`
    $ordinal `,`
    `bindings` `=` `[` $bindings `]`
    (`,` `flags` `=` $flags^)?
    `>`
  }];
}

def IREEInput_PipelineLayoutAttr :
    AttrDef<IREEInput_Dialect, "PipelineLayout", []> {
  let mnemonic = "pipeline.layout";
  let summary = [{executable entry point layout specification}];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$pushConstants,
    ArrayRefParameter<"DescriptorSetLayoutAttr", "">:$setLayouts
  );

  let assemblyFormat = [{
    `<`
    `push_constants` `=` $pushConstants `,`
    `sets` `=` `[` $setLayouts `]`
    `>`
  }];
}

#endif // IREE_DIALECTS_DIALECT_INPUT_BASE_TD
