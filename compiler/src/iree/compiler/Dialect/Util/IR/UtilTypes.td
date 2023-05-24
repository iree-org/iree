// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_TYPES
#define IREE_DIALECT_UTIL_IR_UTIL_TYPES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"

//===----------------------------------------------------------------------===//
// !util.buffer
//===----------------------------------------------------------------------===//

def Util_BufferType : TypeDef<Util_Dialect, "Buffer", [
  Util_ReferenceType,
  Util_SizeAwareType,
  DeclareTypeInterfaceMethods<Util_GlobalType, [
    "isAccessStorageCompatible",
  ]>,
  DeclareTypeInterfaceMethods<Util_InferTypeSize, [
    "inferSizeFromValue",
  ]>,
  DeclareTypeInterfaceMethods<Util_SubrangeType, [
    "createSubrangeOp",
  ]>,
]> {
  let mnemonic = "buffer";

  let summary = [{a reference counted byte buffer}];
  let description = [{
    A reference counted byte buffer that models a pointer, offset, and length.
  }];

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// !util.list<T>
//===----------------------------------------------------------------------===//

def Util_ListType : TypeDef<Util_Dialect, "List"> {
  let mnemonic = "list";

  let summary = [{dense list container type}];
  let description = [{
    Typed container supporting variant storage.
  }];

  let parameters = (ins
    "Type":$element_type
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$element_type
    ), [{
      return $_get(element_type.getContext(), element_type);
    }]>,
  ];

  let assemblyFormat = "`<` custom<ListElementType>($element_type) `>`";

  let extraClassDeclaration = [{
    /// Returns true if the given type can be wrapped in a list.
    static bool isCompatible(Type type);

    /// Returns true if |from| can be implicitly cast to |to| as part of a list
    /// access operation. Example: tensor<*xf32> -> tensor<4xf32>.
    static bool canImplicitlyCast(Type from, Type to);
  }];

  let genVerifyDecl = 1;
}

def Util_AnyListType : TypeAlias<Util_ListType>;

class Util_ListOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::Util::ListType>()">,
      SubstLeaves<"$_self",
                  "$_self.cast<IREE::Util::ListType>().getElementType()",
                  type.predicate>
    ]>, "list<" # type.summary # ">"> {
  string builderCall = !if(!empty(type.builderCall),
                           "", "IREE::Util::ListType::get(" # type.builderCall # ")");
}

//===----------------------------------------------------------------------===//
// !util.ptr<T>
//===----------------------------------------------------------------------===//

def Util_PtrType : TypeDef<Util_Dialect, "Ptr"> {
  let mnemonic = "ptr";

  let summary = [{a pointer-like reference}];
  let description = [{
    A typed indirect reference to a value. These define a runtime addressable
    value that is strongly referenced.
  }];

  let parameters = (ins
    "Type":$target_type
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$target_type
    ), [{
      return $_get(target_type.getContext(), target_type);
    }]>,
  ];

  let assemblyFormat = "`<` qualified($target_type) `>`";

  let extraClassDeclaration = [{
    /// Returns true if the given type can be wrapped in a pointer.
    static bool isCompatible(Type type);
  }];

  let genVerifyDecl = 1;
}

def Util_AnyPtrType : TypeAlias<Util_PtrType>;

class Util_PtrOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::Util::PtrType>()">,
      SubstLeaves<"$_self", "$_self.cast<IREE::Util::PtrType>().getTargetType()",
                  type.predicate>
    ]>, "ptr<" # type.summary # ">"> {
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

def Util_AnyGlobalPtr : TypeAlias<Util_AnyPtrType>;
// TODO(benvanik): actually implement Util_GlobalPtrOf.
class Util_GlobalPtrOf<list<Type> types> : TypeAlias<Util_AnyPtrType> {
  // Suppress unused template argument warning.
  int unused = !size(types);
}

//===----------------------------------------------------------------------===//
// !util.object
//===----------------------------------------------------------------------===//

def Util_ObjectType : TypeDef<Util_Dialect, "Object"> {
  let mnemonic = "object";

  let summary = [{a placeholder for an unspecified object type}];
  let description = [{
    Describes a runtime object type. These may be reference counted or garbage
    collected at runtime.
  }];

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns true if the given type is object-like.
    static bool isCompatible(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// !util.variant
//===----------------------------------------------------------------------===//

def Util_VariantType : TypeDef<Util_Dialect, "Variant"> {
  let mnemonic = "variant";

  let summary = [{a placeholder for a variant type (`?`)}];
  let description = [{
    Describes a runtime variant type. These may be primitives (i32, f32, etc) or
    object types.
  }];

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>
  ];
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_TYPES
