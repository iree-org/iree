// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_DIALECT_TD
#define IREE_DIALECTS_DIALECT_INPUT_DIALECT_TD

include "iree-dialects/Dialect/Input/InputBase.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def IREEInput_BufferViewType : IREEInput_Type<"BufferView"> {
  let mnemonic = "buffer_view";

  let summary = "View into a buffer, with runtime shape and element type";

  let description = [{
    BufferViews represent views onto backing IREE runtime Buffer objects,
    adding runtime shape and element type parameters to the backing buffer.
    BufferViews are typically accepted and returned at boundaries with
    external code.

    In the runtime and lower level compiler, BufferView's are fully modeled;
    however, as boundary types, not all features are exposed publicly. Since
    within compiled tensor programs, it is typical to operate in terms of
    fully typed tensors, the primary mechanism for getting or using a
    BufferView at the high level is by casting to/from a tensor. It is left
    to higher level code to ensure that aliasing rules are enforced at such
    boundaries.
  }];
}

def IREEInput_VariantType : IREEInput_Type<"Variant"> {
  let mnemonic = "variant";

  let summary = "Represents any legal or reference type in the IREE runtime";

  let description = [{
    The variant type is typically used to parameterize container types that
    can contain any legal primitive, reference or null in the IREE type system.
  }];
}

def IREEInput_ListType : IREEInput_Type<"List"> {
  let mnemonic = "list";

  let summary = "A one dimensional list of runtime values";

  let description = [{
    Represents a list of arbitrary type. Primitive types can be expected to
    be efficiently stored in an unboxed form. Reference types and variants
    are permitted.

    Lists can either be homogenous, with a fixed element type, or heterogenous
    by parameterizing them with a VariantType.
  }];

  let parameters = (ins IREEInput_ElementTypeParameter:$elementType);

  let hasCustomAssemblyFormat = 1;
}

def IREEInput_PtrType : IREEInput_Type<"Ptr"> {
  let mnemonic = "ptr";

  let summary = "Pointer to a concrete type";
  let parameters = (ins IREEInput_PtrTargetTypeParameter:$targetType);

  let hasCustomAssemblyFormat = 1;
}

#endif // IREE_DIALECTS_DIALECT_INPUT_DIALECT_TD
