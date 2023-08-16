// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_ATTRS
#define IREE_DIALECT_UTIL_IR_UTIL_ATTRS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// #util.byte_pattern
//===----------------------------------------------------------------------===//

def Util_BytePatternAttr : AttrDef<Util_Dialect, "BytePattern", [
  TypedAttrInterface,
  DeclareAttrInterfaceMethods<Util_SerializableAttrInterface, [
    "getStorageSize",
    "serializeToBuffer",
    "serializeToStream",
  ]>,
]> {
  let mnemonic = "byte_pattern";
  let summary = [{an attribute containing a filled byte pattern}];
  let description = [{
    A dense serializable attribute with the given byte pattern.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"">:$type,
    AttrParameter<"int64_t", "">:$pattern
  );

  let assemblyFormat = [{
    `<` $pattern `>`
  }];
}

//===----------------------------------------------------------------------===//
// #util.byte_range
//===----------------------------------------------------------------------===//

def Util_ByteRangeAttr : AttrDef<Util_Dialect, "ByteRange", []> {
  let mnemonic = "byte_range";
  let summary = [{defines a range of bytes}];
  let description = [{
    Specifies a starting offset and total length in bytes.
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$offset,
    AttrParameter<"int64_t", "">:$length
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// #util.composite
//===----------------------------------------------------------------------===//

def Util_CompositeAttr : AttrDef<Util_Dialect, "Composite", [
  DeclareAttrInterfaceMethods<Util_SerializableAttrInterface, [
    "serializeToBuffer",
    "serializeToStream",
  ]>,
]> {
  let mnemonic = "composite";
  let summary = [{an attribute composed of a sequence of attributes}];
  let description = [{
    Models a concatenated set of serializable attributes that when combined
    form a single sequence of i8 elements. As each value references the uniqued
    storage of the composite element this attribute is cheap to construct. When
    the full flattened range is required it can be efficiently streamed via the
    SerializableAttrInterface. All values must also be serializable.

    All values are tightly packed to byte boundaries. If padding is required it
    can be inserted as splat elements attributes with the padding value
    (usually 0). Sub-byte aligned element types will have their individual
    components padded to byte alignment.
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$totalLength,
    AttrParameter<"ArrayAttr", "">:$values
  );

  let builders = [
    AttrBuilder<(ins
      "ArrayRef<Attribute>":$values
    )>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumElements() const { return getTotalLength(); }
  }];

  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

def Util_CASScope : AttrDef<Util_Dialect, "CASScope"> {
  let mnemonic = "cas_scope";
  let summary = "A unique scope delimitter for bindings CAS resources";
  let description = [{
    CAS resources can be globally (context bound) or locally (sub-tree)
    bound. Locally scoped resources are eligible for various lifetime
    policies, liveness analysis, etc, and they can be turned into tombstones
    if it is deemed valid to do so (releasing the memory they hold while 
    keeping them in index so as to not produce dangling pointers).

    A scope is demarcated by a #util.cas-scope attribute, which internally
    keeps a process unique number for each unique attribute created (either
    by parsing or builder). This will typically be at the module level as:

    ```
    module attributes {util.cas-scope = #util.cas-scope} {

    }
    ```

    Resource management APIs will typically walk the hierarchy when a local
    scope is requested and will find/create these attributes as needed.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$uniqueIndex
  );
  let builders = [
    // Builds with a unique global for the process.
    AttrBuilder<(ins)>,
  ];
  let extraClassDeclaration = [{
    // Looks for a `util.cas-scope = #util.cas-scope` attribute at each level
    // of the hierarchy, starting with `op`, returning the first found. If none
    // is found, then create one and attach it to the outermost operation.
    static CASScopeAttr findOrCreateRootScope(Operation *op);
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Util_CASElementsAttr : AttrDef<Util_Dialect, "CASElements", [
    ElementsAttrInterface,
    DeclareAttrInterfaceMethods<Util_SerializableAttrInterface, [
      "serializeToBuffer",
      "serializeToStream",
    ]>,
]> {
  let mnemonic = "elements";
  let summary = "An ElementsAttr which uses scoped, interned blob resources";
  let description = [{
    The `#util.elements` attribute is an ElementsAttr implementation that:

    * Presumes that its backing data is stored in IREE's serialization form
      directly without translation.
    * Is backed by blob resources managed out-of-line with the IR as dialect
      resources.
    * Interops with the `CASManagerDialectInterface` to provide local/global
      scoping to managed data, which enables policies of GC'ing the backing
      data as constraints allow during compilation.
    * Is intended to be an efficient option within the compiler for producing
      transformed elements (i.e. via constant evaluation, folding, etc)
      with the ability to trim unreferenced data.
    * Interns/uniques within scopes by managing hash and liveness metadata
      with the blob.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"", "ShapedType">:$type,
    ResourceHandleParameter<"UtilDialect::CASResourceHandle">:$handle
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "ShapedType":$type, "UtilDialect::CASResourceHandle":$handle
    )>,
  ];
  let extraClassDeclaration = [{
    // Gets the backing resource data. If the data cannot be resolved, or
    // is not valid, an error is emitted.
    // The returned data is valid so long as the resource is alive.
    std::optional<CASResourceReader> getResourceData(Location loc) const;

    // Gets the backing resource data, emitting no error if it cannot be found
    // or is not valid.
    // The returned data is valid so long as the resource is alive.
    std::optional<CASResourceReader> getOptionalResourceData() const;

    // ElementsAttr implementation.
    using ContiguousIterableTypesT = std::tuple<
      // Integer types.
      // uint8_t, uint16_t, uint32_t, uint64_t,
      // int8_t, int16_t, int32_t, int64_t,
      // short, unsigned short, int, unsigned, long, unsigned long,
      // // Float types.
      // float, double
    >;
    using NonContiguousIterableTypesT = std::tuple<
      // Integer types.
      APInt,
      // Float types.
      APFloat
    >;

    // template <typename T>
    // auto try_value_begin_impl(OverloadToken<T>) const {
    //   return try_value_begin<T>();
    // }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_ATTRS
